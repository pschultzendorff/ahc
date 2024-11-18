from dataclasses import dataclass, field


@dataclass
class Name:
    name: str = field(default="John")

    def reset(self) -> None:
        self.name = "Jane"


@dataclass
class Age:
    age: int = field(default=30)


@dataclass
class Person(Name, Age):
    pass


person = Person()
person.reset()
