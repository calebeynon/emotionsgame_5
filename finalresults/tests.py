from otree.api import Currency as c, currency_range, expect, Bot, Submission
from . import *


class PlayerBot(Bot):
    def play_round(self):
        yield survey, dict(q1='Male', q2='White or Caucasian', q3=20, q4='Computer Science', q5=1, q6='Important')
        yield Submission(payment, check_html=False)
