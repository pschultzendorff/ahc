class Parent:
    def __init__(self):
        print("parent")

    def foo(self):
        print("foo parent")


class Child(Parent):
    def __init__(self):
        print("child")

    def foo(self):
        print("foo child")


class MixinA:
    def __init__(self):
        print("MixinA")

    def foo(self):
        print("foo mixinA")


class MixinB:
    def __init__(self):
        print("MixinB")

    def foo(self):
        print("foo mixinB")


class Composed(MixinA, MixinB, Child): ...


composed = Composed()
composed.foo()
