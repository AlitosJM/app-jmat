class Post:
    CONST_NUM0 = '''
    Mi nombre es José Miguel Alí Toscano, soy maestro en ciencias en bioelectrónica. Apasionado por
    proyectos relacionados a la ingeniería electrónica, he participado en el desarrollo de diversos proyectos desde hardware y firmware con base en lenguaje C para microntroladores de 8 bits.

    Mi interés general se encuentra en la generación de intrumentación electrónica que sea de utilidad para resolver problemas en específico.

    Actualmente, tengo interés en el diseño y desarrollo de aplicaciones web y machine learning con Python con el fin de brindar soluciones oportunas para satisfacer las necesidades
    empresariales, académicas ó de asistencia tecnológica.

    Estas páginas son parte de mi primer proyecto de diseño web y la he desarrollado en Flask para Python.
    '''

    def __init__(self, post_id, title, subtitle, body):
        self.id = post_id
        self.title = title
        self.subtitle = subtitle
        self.body = body

    # @classmethod
    # def classmethod(cls):
    #     # Haha. I am kidding you
    #     return cls.staticmethod()
    #
    # @staticmethod
    # def staticmethod():
    #     return Post.CONST_NUM0
