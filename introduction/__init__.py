from otree.api import *
class C(BaseConstants):
    NAME_IN_URL = 'intro'
    PLAYERS_PER_GROUP = 4
    NUM_ROUNDS = 1

class Subsession(BaseSubsession):
    pass
class Group(BaseGroup):
    pass
class Player(BasePlayer):
    input_you = models.FloatField()
    input2 = models.FloatField()
    input3 = models.FloatField()
    input4 = models.FloatField()
    result = models.FloatField()

class all_instructions(Page):
    pass
class instructions_x_quiz(Page):
    pass
class practice_only(Page):
    pass
class Intro(Page):
    pass
class p2(Page):
    pass
class p3(Page):
    pass
class p4(Page):
    pass
class p5(Page):
    pass
class p6(Page):
    pass
class practice(Page):
    form_model='player'
    form_fields = ['input_you','input2','input3','input4']
class p7(Page):
    pass
class quiz(Page):
    pass

page_sequence = [instructions_x_quiz,practice_only,all_instructions]
#page_sequence = [Intro,p2,p3,p4,p5,p6,practice,p7,quiz]