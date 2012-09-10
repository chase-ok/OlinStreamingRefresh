#!/usr/bin/env
"""
Contains a set of helper methods/classes to make organization of tasks and
parameters a little easier.
"""

class MissingDependency(Exception):

    def __init__(self, task, dependency):
        self.task = task
        self.dependency = dependency

    def __str__(self):
        return """Missing dependency: task "{0}" is dependent on {1}, but this
               dependency is not declared in {2}.dependencies!""".format(
               str(self.task), str(self.dependency), str(self.task.__class__))

class Task(object):
    """
    Some accomplishable task (e.g. identifying particles) that may or may
    not have dependencies.
    """

    dependencies = []
    name = "<INSERT TASK NAME HERE>"

    def __init__(self, context):
        """Creates a new task to operate inside of the given context.

        This does NOT run the task itself. 
        
        All subclasses should implement a constructor that takes a single 
        argument, context (a Context instance).
        """
        self.context = context

    def isComplete(self):
        """Returns true if this task has already been completed.

        This should be determined by investigating context.
        """ 
        return False

    def run(self):
        """Run this task in the current context.

        This should to be overwritten by subclasses.
        """
        raise NotImplemented()

    def export(self):
        """Returns a dict of exports.

        If this task is exporting any values to other tasks, it should return
        a dict mapping from export name to value.
        """
        return dict()
        
    def _import(self, task, export):
        """Returns the value of an export from the given task class.
        
        This is just a helper method to the current context.
        """
        try:
            exports = self.context.exports[task]
        except KeyError:
            raise MissingDependency(self, task)
        return exports[export]

    def _param(self, param):
        """Shortcut to context.params[param]."""
        return self.context.params[param]

    def __str__(self):
        return self.name

class Logger(object):

    def __init__(self, showDebug=True, showWarnings=True):
        self._showDebug = showDebug
        self._showWarnings = showWarnings

    def log(self, format, *args):
        self._doLog(format.format(*args))

    def debug(self, format, *args):
        if self._showDebug: 
            self.log("DEBUG: " + format, *args)

    def warn(self, format, *args):
        if self._showWarning:
            self.log("WARNING: " + format, *args)

    def _doLog(self, message):
        raise NotImplemented()

class PrintLogger(Logger):

    def _doLog(self, message):
        print message

class Context(object):
    """
    Maintains most of the program state. Stores references to the hdf5 file,
    cached values, parameters, exports, etc.
    """
    
    def __init__(self, hdf5, root, paramValues=None, logger=None):
        """Creates a new context from an hdf5 handle and parameter values.

        hdf5 is a handle to an exisiting hdf5 connection (i.e. from 
        tables.openFile).
        root is the node inside of the hdf5 file that this data set is going
        to run under.
        paramValues is an optional dict of mapping from param names to values
        in order to override their defaults (see _createParamValues).
        logger defaults to PrintLogger.
        """
        self.hdf5 = hdf5
        self.root = root
        self.params = _createParamValues(paramValues)
        self.logger = logger if logger else PrintLogger()
        self.exports = dict()

    @property
    def attrs(self):
        """Returns the attrs of the root hdf5 node."""
        return self.root._v_attrs

    def hasNode(self, name):
        """Returns true if root has a child node with the given name."""
        return hasattr(self.root, name)

    def node(self, name):
        return getattr(self.root, name)

    def clearNode(self, name):
        """Removes the node with the given name from root."""
        if self.hasNode(name):
            getattr(self.root, name)._f_remove(recursive=True)

    def createTable(self, name, *args, **kwargs):
        """Creates a table with the given name under root.

        args and kwargs are passed on to the hdf5 createTable method.
        """
        self.clearNode(name)
        return self.hdf5.createTable(self.root, name, *args, **kwargs)

    def createArray(self, name, *args, **kwargs):
        """Creates an array with the given name under root.

        args and kwargs are passed on to the hdf5 createTable method.
        """
        self.clearNode(name)
        return self.hdf5.createArray(self.root, name, *args, **kwargs)

    def flush(self):
        """Flushes the entire hdf5 connection.
        """
        self.hdf5.flush()

    def addExports(self, task, exports):
        """Merges the exports from the given task into the exports dict."""
        self.exports[task] = exports

    def log(self, *args, **kwargs):
        """Shortcut to logger.log()"""
        self.logger.log(*args, **kwargs)

    def debug(self, *args, **kwargs):
        """Shortcut to logger.debug()"""
        self.logger.debug(*args, **kwargs)

    def warn(self, *args, **kwargs):
        """Shortcut to logger.warn()"""
        self.logger.debug(*args, **kwargs)

class DependencyCycle(Exception):

    def __init__(self, task, unresolvedDependencies):
        self.task = task
        self.unresolved= unresolvedDependencies

    def __str__(self):
        return """There is a dependency cycle: task "{0}" cannot resolve the
               following dependencies: {1}!""".format(
               self.task.name, ", ".join(t.name for t in self.unresolved))

class Scheduler(object):
    """
    Given a set of tasks, this will schedule/execute them subject to their 
    dependencies and availability of data.
    """
    
    def __init__(self, tasks=None):
        """Takes a list of Task classes (order doesn't matter)."""
        self._tasks = set()
        if tasks:
            for task in tasks: self.addTask(task)
        
    def addTask(self, taskClass):
        """Adds a task (and all of its dependencies) to the schedule."""
        if taskClass not in self._tasks:
            self._tasks.add(taskClass)
            for dependency in taskClass.dependencies:
                self.addTask(dependency)
        
    def run(self, context, forceRedo=False):
        """Runs all of the scheduled tasks under the given context.

        forceRedo=True forces all of the tasks to be run, regardless of any
        cached values (useful for parameter changes)."""
        # forward = dependent on, reverse = pre-req of 
        forward, reverse = self._computeLinks()

        freeTasks = [task for task, deps in forward.iteritems() if not deps]
        schedule = []
        while freeTasks:
            task = freeTasks.pop()
            schedule.append(task)

            for dependentTask in reverse[task]:
                forward[dependentTask].remove(task)
                if not forward[dependentTask]:
                    freeTasks.append(dependentTask)

        for task, dependencies in forward.iteritems():
            if dependencies:
                raise DependencyCycle(task, dependencies)

        context.debug("Schedule={0}", [t.name for t in schedule])
        for task in schedule:
            self._runTask(task, context, forceRedo)

    def _computeLinks(self):
        forward = dict((task, set(task.dependencies)) for task in self._tasks)

        reverse = dict((task, set()) for task in self._tasks)
        for task in self._tasks:
            for dependency in task.dependencies:
                reverse[dependency].add(task)

        return forward, reverse

    def _runTask(self, taskClass, context, forceRedo):
        task = taskClass(context)

        context.log("Starting task: {0}", taskClass.name)
        if forceRedo or not task.isComplete():
            task.run()
        else:
            context.log("Already complete.")
        context.log("Finished task: {0}", taskClass.name)

        context.addExports(taskClass, task.export())

class Parameter(object):
    """
    Some (constant) parameter for task execution that can be identified by a
    global name.
    """

    def __init__(self, name, defaultValue, description):
        """Creates a new parameter (should NOT be called directly).

        name and description are strings.
        default value can be anything.
        """
        self.name = name
        self.defaultValue = defaultValue
        self.description = description

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Parameter) and self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Parameter({0}, {1}, {2})".format(
               repr(self.name), repr(self.defaultValue), repr(self.description))

# global container holding parameter instances (indexed by name)
_registeredParams = dict()

def registerParameter(name, defaultValue=None, description=""):
    """Registers a param with the given name, defaultValue, and description.

    Returns a Parameter instance that can be used to access its value from an
    Hdf5Data instance.
    """
    if name in _registeredParams:
        raise ValueError("Parameter {0} is registered twice. Perhaps there " +
                         "are two different usages?".format(name))

    param = Parameter(name, defaultValue, description)
    _registeredParams[name] = param
    return param

def _createParamValues(nameSpecifiedValues=None):
    """Merges default and user-specified paramter values.

    nameSpecifiedValues should be a dict mapping from name to paramter value
    (does not need to include all possible parameters, just those that need to
    have a non-default value).
    Returns a dict mapping from Parameter instance to value.
    """
    values = dict((p, p.defaultValue) for p in _registeredParams.values())
    
    if nameSpecifiedValues:
        for name, value in nameSpecifiedValues.iteritems():
            if name not in _registeredParams:
                raise ValueError("No such parameter {0}!".format(name))
            values[_registeredParams[name]] = value
    
    return values


