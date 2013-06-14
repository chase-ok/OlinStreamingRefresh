#!/usr/bin/env
"""
Contains a set of helper methods/classes to make organization of tasks and
parameters a little easier.
"""

import tables as tb
import numpy as np

class MissingDependency(Exception):
    def __init__(self, task, dependency):
        self.task = task
        self.dependency = dependency
    def __str__(self):
        return """Missing dependency: task "{0}" is dependent on {1}, but this
               dependency is not declared in {2}.dependencies!""".format(
               self.task, self.dependency, self.task.__class__)

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


class MissingExport(Exception):
    def __init__(self, interface, task, export):
        self.interface = interface
        self.task = task
        self.export = export
    def __str__(self):
        return """{0} did not export {1} as promised! {0} implements
               interface {2} and therefore must export {1}.""".format(
               self.task, self.export, self.interface)

class TaskInterface(Task):

    willExport = []

    def __init__(self,  implementation, context):
        super(TaskInterface, self).__init__(context)
        self._impl = implementation

    def isComplete(self): return self._impl.isComplete()
    def run(self): self._impl.run()

    def export(self):
        exports = self._impl.export()
        for export in self.willExport:
            if export not in exports:
                raise MissingExport(self, self._impl, export)
        return exports


class MultipleDefaultImplementations(Exception):
    def __init__(self, interface, tasks):
        self.interface = interface
        self.task = tasks
    def __str__(self):
        return "{0} has multiple default implementations: {1} (at least)."\
               .format(self.interface, ", ".join(map(str,self.tasks)))

_implementations = dict()
_defaultImplementations = dict()
def implements(task, interface, default=False):
    _implementations.setdefault(interface, set()).add(task)
    if default:
        if interface in _defaultImplementations:
            other = _defaultImplementations[interface]
            raise MultipleDefaultImplementations(self, [task, other])
        _defaultImplementations[interface] = task


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

DEFAULT_FILTERS = tb.Filters(complevel=1, complib='blosc')

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
        kwargs.setdefault('filters', DEFAULT_FILTERS)
        return self.hdf5.createTable(self.root, name, *args, **kwargs)

    def createChunkArray(self, name, array, *args, **kwargs):
        """Creates a chunked array with the given name under root.

        args and kwargs are passed on to the hdf5 createCArray method.
        """
        self.clearNode(name)
        kwargs.setdefault('filters', DEFAULT_FILTERS)
        kwargs.setdefault('atom', tb.Atom.from_dtype(array.dtype))
        kwargs.setdefault('shape', array.shape)
        cArray = self.hdf5.createCArray(self.root, name, *args, **kwargs)
        cArray[:, :] = array
        return cArray

    def createArray(self, name, *args, **kwargs):
        """Creates an array with the given name under root.

        args and kwargs are passed on to the hdf5 createArray method.
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
               self.task, ", ".join(t.name for t in self.unresolved))

class MissingImplementation(Exception):
    def __init__(self, interface):
        self.interface = interface
    def __str__(self):
        return """No implementation of {0} was included in the schedule and no 
               default was found.""".format(self.interface)

class MultipleImplementations(Exception):
    def __init__(self, interface, tasks):
        self.interface = interface
        self.task = tasks
    def __str__(self):
        return """{0} has multiple implementations in the schedule: {1}. There 
               must be one (or zero if a default implementation is 
               specified).""".format(
               self.interface, ", ".join(map(str, self.tasks)))

class Scheduler(object):
    """
    Given a set of tasks, this will schedule/execute them subject to their 
    dependencies and availability of data.
    """
    
    def __init__(self, tasks=None):
        """Takes a list of Task classes (order doesn't matter)."""
        self._tasks = set()
        self._redos = set()
        if tasks:
            for task in tasks: self.addTask(task)
        
    def addTask(self, taskClass, forceRedo=False):
        """Adds a task (and all of its dependencies) to the schedule.

        If forceRedo is True, the task and all those dependent on it will be 
        recomputed."""
        if taskClass not in self._tasks:
            self._tasks.add(taskClass)
            for dependency in taskClass.dependencies:
                self.addTask(dependency)
        if forceRedo:
            self._redos.add(taskClass)
        
    def run(self, context, forceRedo=False):
        """Runs all of the scheduled tasks under the given context.

        forceRedo=True forces all of the tasks to be run, regardless of any
        cached values (useful for parameter changes)."""
        implementations = self._fillInterfaces()

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

        #context.debug("Schedule={0}", [t.name for t in schedule])
        recomputed = set()
        for i, taskClass in enumerate(schedule):
            if issubclass(taskClass, TaskInterface):
                impl = implementations[taskClass](context)
                task = taskClass(impl, context)
            else:
                task = taskClass(context)

            self._runTask(task, context, forceRedo, recomputed)
            self._resolveExports(task, context, schedule[i+1:])

    def _fillInterfaces(self):
        to_add = []
        implementations = {}

        for interface in self._tasks:
            if not issubclass(interface, TaskInterface): continue

            allImpls = [impl for impl in _implementations[interface]
                        if impl in self._tasks]
            if len(allImpls) == 0:
                try:
                    impl = _defaultImplementations[interface]
                    if impl not in self._tasks: to_add.append(impl)
                except KeyError:
                    raise MissingImplementation(interface)
            elif len(allImpls) == 1:
                impl = allImpls[0]
            else:
                raise MultipleImplementations(interface, allImpls)

            interface.dependencies.append(impl)
            implementations[interface] = impl

        for task in to_add: self.addTask(task)
        return implementations

    def _computeLinks(self):
        forward = dict((task, set(task.dependencies)) for task in self._tasks)

        reverse = dict((task, set()) for task in self._tasks)
        for task in self._tasks:
            for dependency in task.dependencies:
                reverse[dependency].add(task)

        return forward, reverse

    def _runTask(self, task, context, forceRedo, recomputed):
        context.log("Starting task: {0}", task.name)
        if forceRedo \
                or task.__class__ in self._redos \
                or any(dep in recomputed for dep in task.dependencies) \
                or not task.isComplete():
            task.run()
            recomputed.add(task.__class__)
        else:
            context.log("Already complete.")
        context.log("Finished task: {0}", task.name)

    def _resolveExports(self, task, context, remaining):
        def stillNeeded(taskClass):
            for laterTask in remaining:
                if taskClass in laterTask.dependencies:
                    return True
            return False

        if stillNeeded(task.__class__):
            context.addExports(task.__class__, task.export())

        # see if we can free up some memory
        for dependency in task.dependencies:
            if not stillNeeded(dependency):
                try:
                    del context.exports[dependency]
                except KeyError: 
                    pass


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


