from otree.api import *


class C(BaseConstants):
    NAME_IN_URL = 'supergame1'
    PLAYERS_PER_GROUP = 4
    NUM_ROUNDS = 3
    ENDOWMENT = cu(25)
    MULTIPLIER = 0.4


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    total_contribution = models.CurrencyField()
    individual_share = models.CurrencyField()


class Player(BasePlayer):
    contribution = models.CurrencyField(
        min=0, max=C.ENDOWMENT, label="How many tokens will you place in the group account?"
    )


# FUNCTIONS
def creating_session(subsession: Subsession):
    grouping = [[1,5,9,13],[2,6,10,14],[3,7,11,15],[4,8,12,16]]
    subsession.set_group_matrix(grouping)

def set_payoffs(group: Group):
    players = group.get_players()
    contributions = [p.contribution for p in players]
    group.total_contribution = sum(contributions)
    group.individual_share = (
        group.total_contribution * C.MULTIPLIER
    )
    for p in players:
        p.payoff = C.ENDOWMENT - p.contribution + group.individual_share
        
        if 'payoff_list' not in p.participant.vars:
            p.participant.vars['payoff_list'] = []
        
        p.participant.vars['payoff_list'].append(p.payoff)
        p.participant.vars['payoff_sum_1'] = sum(p.participant.vars['payoff_list'])
        
def set_payoffs_0(group: Group):
    players = group.get_players()
    for p in players:
        p.participant.vars['payoff_list'] = []


# PAGES
class StartPage(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number==1
    @staticmethod
    def vars_for_template(player):
        players = player.get_others_in_group()
        labels = [p.participant.label for p in players]
        return {
            'one':labels[0],
            'two':labels[1],
            'three':labels[2]
            }
        
class RoundWaitPage(WaitPage):
    pass

class ChatFirst(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1
    
    @staticmethod
    def get_timeout_seconds(player):
        return 120

class Chat(Page):
    @staticmethod
    def vars_for_template(player):
        if player.round_number >1:
            players = player.get_others_in_group()
            prevs = [p.in_previous_rounds() for p in players]
            labels = [p.participant.label for p in players]
            prev = player.in_previous_rounds()
            conts = [];labs = []
            for p,l in zip(prevs,labels):
                rec = p[-1]
                conts.append(rec.contribution)
                labs.append(l)
            mycont = prev[-1].contribution
            mypay = prev[-1].payoff
            return {
                'lab1':labs[0],
                'lab2':labs[1],
                'lab3':labs[2],
                'cont1':conts[0],
                'cont2':conts[1],
                'cont3':conts[2],
                'mycont':mycont,
                'mypay':mypay
                }
        else:
            return {
                'lab1':None,
                'lab2':None,
                'lab3':None,
                'cont1':None,
                'cont2':None,
                'cont3':None,
                'mycont':None,
                'mypay':None
                }
    timeout_seconds = 30
    
    @staticmethod
    def is_displayed(player):
        return player.round_number > 1

class ChatWaitPage(WaitPage):
    pass

class Contribute(Page):
    form_model = 'player'
    form_fields = ['contribution']


class ResultsWaitPage(WaitPage):
    after_all_players_arrive = set_payoffs


class ResultsOnly(Page):
    @staticmethod
    def vars_for_template(player):
        # Get all players in current group (including self) ordered by id_in_group
        players = sorted(player.group.get_players(), key=lambda p: p.id_in_group)
        labels = [p.participant.label for p in players]
    
        # Build table: one row per round, with each player's contribution and payoff
        table_data = []
        for r in range(1, player.round_number + 1):
            round_players = player.in_round(r).group.get_players()
            round_players_sorted = sorted(round_players, key=lambda p: p.id_in_group)
    
            row = {
                'round': r,
                'contributions': [int(p.in_round(r).contribution) for p in round_players_sorted],
                'payoffs': [round(float(p.in_round(r).payoff),2) for p in round_players_sorted]
            }
            table_data.append(row)
    
        conts = [p.contribution for p in players if p != player]
        payoffs = [p.payoff for p in players if p != player]
        labels_others = [p.participant.label for p in players if p != player]
    
        return {
            'pay': round(float(player.payoff),2),
            'priv': int(25 - player.contribution),
            'one_cont': int(conts[0]),
            'two_cont': int(conts[1]),
            'three_cont': int(conts[2]),
            'one': labels_others[0],
            'two': labels_others[1],
            'three': labels_others[2],
            'pcont': int(player.contribution),
            'gcont': int(player.group.total_contribution),
            'player_labels': labels,
            'table_data': table_data,
            'pay_share': round(float(player.group.individual_share),2)
        }
    @staticmethod
    def get_timeout_seconds(player):
        return 25

class Results(Page):
    @staticmethod
    def vars_for_template(player):
        # Get all players in current group (including self) ordered by id_in_group
        players = sorted(player.group.get_players(), key=lambda p: p.id_in_group)
        labels = [p.participant.label for p in players]
        my_label = player.participant.label

        # Build table: one row per round, with each player's contribution and payoff
        table_data = []
        for r in range(1, player.round_number + 1):
            round_players = player.in_round(r).group.get_players()
            round_players_sorted = sorted(round_players, key=lambda p: p.id_in_group)
    
            row = {
                'round': r,
                'contributions': [int(p.in_round(r).contribution) for p in round_players_sorted],
                'payoffs': [round(float(p.in_round(r).payoff),2) for p in round_players_sorted]
            }
            table_data.append(row)
    
        conts = [p.contribution for p in players if p != player]
        payoffs = [p.payoff for p in players if p != player]
        labels_others = [p.participant.label for p in players if p != player]
    
        return {
            'pay': round(float(player.payoff),2),
            'priv': int(25 - player.contribution),
            'one_cont': int(conts[0]),
            'two_cont': int(conts[1]),
            'three_cont': int(conts[2]),
            'my_label': my_label,
            'one': labels_others[0],
            'two': labels_others[1],
            'three': labels_others[2],
            'pcont': int(player.contribution),
            'gcont': int(player.group.total_contribution),
            'player_labels': labels,
            'table_data': table_data,
            'pay_share': round(float(player.group.individual_share),2)
        }
    @staticmethod
    def get_timeout_seconds(player):
        if player.round_number == 1:
            return 90
        else:
            return 60
    
class RegroupingMessage(Page):
    @staticmethod
    def is_displayed(player):
        return player.round_number == C.NUM_ROUNDS
    @staticmethod
    def before_next_page(player,timeout_happened):
        set_payoffs_0(player.group)
    @staticmethod
    def vars_for_template(player):
        return {
            "total_payoff": player.participant.vars.get('payoff_sum_1',0),
            "message": "You are now being regrouped. Please wait while we reassign groups."
        }



page_sequence = [StartPage,RoundWaitPage,Contribute, ResultsWaitPage, ResultsOnly, Results, RegroupingMessage]
