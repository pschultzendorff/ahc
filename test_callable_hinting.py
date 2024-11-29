from typing import Callable, Optional


def foo(arg: Optional[int] = None) -> float:
    if arg is None:
        return 0.1
    else:
        return arg + 0.1


# This gives the mypy error "Too few arguments".
class Foo:

    def foo(
        self,
        a: Optional[int] = None,
    ) -> int:
        if a is not None:
            return a
        else:
            return 0


class Bar:

    foo: Callable[[Optional[int]], float]

    def bar(self) -> None:
        print(self.foo(1))


class FooBar(Foo, Bar): ...
