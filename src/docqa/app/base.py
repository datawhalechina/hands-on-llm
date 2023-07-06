class Register:

    _dict = {}

    @classmethod
    def regist(cls, _class):
        cls._dict[_class.__name__] = _class
        return _class

    @classmethod
    def get(cls, name: str, typ: str = "llm"):
        cls_name = name.title() + typ.title()
        return Register._dict.get(cls_name)
