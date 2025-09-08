from otree.api import Currency as c, currency_range, expect, Bot
from . import *


class PlayerBot(Bot):
    def play_round(self):
        yield chatExplanation
        yield Chat
        yield RegroupingMessage
