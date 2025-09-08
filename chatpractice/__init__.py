from otree.api import *



class C(BaseConstants):
    NAME_IN_URL = 'chatpractice'
    PLAYERS_PER_GROUP = 4
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    total_contribution = models.CurrencyField()
    individual_share = models.CurrencyField()


class Player(BasePlayer):
    pass


# FUNCTIONS

# PAGES
class chatExplanation(Page):
    pass
class RoundWaitPage(WaitPage):
    pass

class Chat(Page):
    timeout_seconds = 60

class ChatWaitPage(WaitPage):
    pass

class RegroupingMessage(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == C.NUM_ROUNDS
    @staticmethod
    def vars_for_template(player):
        return {
            "message": "You are now being regrouped. Please wait while we reassign groups."
        }

# =============================================================================
# class SwitchApps(WaitPage):
#     wait_for_all_groups = True
#     @staticmethod
#     def is_displayed(player):
#         return player.round_number == C.NUM_ROUNDS
#     
#     @staticmethod
#     def after_all_players_arrive(subsession):
#         new_structure = [[1,5,9,12],[2,6,10,8],[3,4,7,11]]
#         players = subsession.get_players()
#         id_to_player = {p.id_in_subsession: p for p in players}
# 
#         new_groups = [[id_to_player[i] for i in group if i in id_to_player] for group in new_structure]
#         
#         # Store the new group structure in participant.vars so it persists
#         for group in new_groups:
#             for player in group:
#                 player.participant.vars['new_group'] = [p.id_in_subsession for p in group]
#         
#         subsession.set_group_matrix(new_groups)s
# =============================================================================

page_sequence = [chatExplanation,RoundWaitPage,Chat,ChatWaitPage,RegroupingMessage]
