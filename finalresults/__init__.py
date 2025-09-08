from otree.api import *
class C(BaseConstants):
    NAME_IN_URL = 'final'
    PLAYERS_PER_GROUP = 4
    NUM_ROUNDS = 1

class Subsession(BaseSubsession):
    pass
class Group(BaseGroup):
    pass
class Player(BasePlayer):
    q1 = models.StringField(choices=['Male', 'Female', 'Other/Prefer not to respond'], widget=widgets.RadioSelect,
                            label="What gender do you identify as?")
    q2 = models.StringField(
        choices=[
            'White or Caucasian',
            'Black or African American',
            'Hispanic or Latino/a/x',
            'Asian or Asian American',
            'Native American or Alaska Native',
            'Middle Eastern or North African',
            'Native Hawaiian or Other Pacific Islander',
            'Multiracial',
            'Prefer not to say'
        ],
        widget=widgets.RadioSelect, label="What is your ethnicity?"
    )
    q3 = models.IntegerField(label = "Your Age:")
    q4 = models.StringField(label = "Your Major:")
    q5 = models.IntegerField(label = "How many siblings did you have growing up?")
    q6 = models.StringField(choices = [
        'Not at all important',
        'Slightly important',
        'Important',
        'Fairly important',
        'Very important'
        ],widget = widgets.RadioSelect,label = 'How important is religion in your daily life?')
    final_payoff = models.CurrencyField()
class payment(Page):
    @staticmethod
    def vars_for_template(player):
        session = player.session
        p = player.participant.payoff + cu(75)
        player.final_payoff = p
        return {
            'pay': int(player.participant.payoff),
            'payoff': p.to_real_world_currency(session),
            'segment_chosen': player.participant.vars['seg_chosen']
            }
class survey(Page):
    form_model = 'player'
    form_fields = ['q1','q2','q3','q4','q5','q6']
page_sequence = [survey, payment]