class Parent:
    def __init__(self):
        print("parent")


class Child(Parent):
    def __init__(self):
        print("child")
        super().__init__()


class MixinA:
    def __init__(self):
        print("MixinA")
        super().__init__()


class MixinB:
    def __init__(self):
        print("MixinB")
        super().__init__()


class Composed(MixinA, MixinB, Child): ...


composed = Composed()
