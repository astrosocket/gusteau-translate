from abc import abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator
from heapq import heapify, heappop, heappush
from typing import Any, Protocol, Self, cast, runtime_checkable

# The majority of this module is pulled from Dawid Kraczkowski's
# [crafting a data-processing pipeline article](https://github.com/dkraczkowski/dkraczkowski.github.io/tree/main/articles/crafting-data-processing-pipeline)
# with modifications due to updates in python's typing system


class PipelineError(Exception): ...


type NextStep[Context] = Callable[[Context], Iterable[Exception | Context]]
type ErrorHandler[Context] = Callable[
    [Exception, Context, PipelineStep, NextStep], None
]


@runtime_checkable
class PipelineStep[Context](Protocol):
    priority: int = 0

    # TODO: need comparison method for priority queue
    @abstractmethod
    def __call__(self, context: Context, next_step: NextStep) -> Generator[Context]: ...

    def _is_valid_operand(self, other: Self) -> bool:
        return hasattr(other, "priority")

    def __lt__(self, other: Self) -> bool:
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.priority < other.priority

    def __le__(self, other: Self) -> bool:
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.priority <= other.priority

    def __gt__(self, other: Self) -> bool:
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.priority > other.priority

    def __ge__(self, other: Self) -> bool:
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.priority >= other.priority

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, PipelineStep[Context]) and self.priority == other.priority
        )


type PipelineQueue[PipelineStep] = list[PipelineStep]


def _default_error_handler[Context](
    error: Exception,
    context: Context,
    current_step: PipelineStep,
    next_step: NextStep,
):
    raise error


class PipelineCursor[Context]:
    queue: PipelineQueue[PipelineStep[Context]]
    """Queue of steps to take in the pipeline"""
    error_handler: ErrorHandler
    """Callable to handle errors"""

    def __init__(
        self, steps: PipelineQueue[PipelineStep[Context]], error_handler: ErrorHandler
    ):
        self.queue = steps
        self.error_handler = error_handler

    def __call__(self, context: Context):
        if not self.queue:
            return
        current_step = heappop(self.queue)
        next_step = PipelineCursor[Context](self.queue, self.error_handler)
        try:
            current_step(context, next_step)
        except Exception as error:
            self.error_handler(error, context, current_step, next_step)

    def __repr__(self):
        return f"PipelineCursor({[s.__class__.__name__ for s in self.queue]})"


class Pipeline[Context]:
    queue: PipelineQueue[PipelineStep[Context]]

    def __init__(self, *steps: PipelineStep[Context]):
        steps_list = list(steps)
        heapify(steps_list)

        self.queue = cast(PipelineQueue[PipelineStep[Context]], steps_list)

    def append(self, step: PipelineStep[Context]):
        heappush(self.queue, step)

    def extend(self, steps: Iterable[PipelineStep[Context]]):
        for step in steps:
            self.append(step)

    def remove(self, step_name: str) -> PipelineStep[Context] | None:
        found_step = None
        for step in self.queue:
            if step_name == step.__class__.__name__:
                found_step = step
                break
        if found_step is not None:
            self.queue.remove(found_step)
            heapify(self.queue)
        return found_step

    def __iter__(self) -> Iterator[PipelineStep[Context]]:
        return iter(self.queue)

    def __call__(self, context: Context, error_handler: ErrorHandler | None = None):
        execute = PipelineCursor[Context](
            self.queue, error_handler or _default_error_handler
        )
        execute(context)

    def __len__(self) -> int:
        return len(self.queue)

    def __repr__(self) -> str:
        r = f"{self.__class__.__name__}=["
        if self.queue:
            for step in self.queue[:-1]:
                r += f"{step!r}, "
            r += f"{self.queue[-1]!r}"
        r += "]"
        return r
