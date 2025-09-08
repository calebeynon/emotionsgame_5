from otree.api import Currency as c, currency_range, expect, Bot, Submission
from . import *


class PlayerBot(Bot):
    def play_round(self):
        if self.player.round_number == 1:
            yield StartPage
        yield Contribute, dict(contribution=10)
        yield Submission(Results, check_html=False)
        if self.player.round_number == C.NUM_ROUNDS:
            yield RegroupingMessage
