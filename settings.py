from os import environ

SESSION_CONFIGS = [
    dict(
        name = 'public_goods',
        app_sequence = ['introduction','supergame1','supergame2','supergame3','supergame4','supergame5','finalresults'],
        num_demo_participants = 16,
        room = 'room'
        ),
]

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=0.1, participation_fee=7.5
)

PARTICIPANT_FIELDS = []
SESSION_FIELDS = []
LANGUAGE_CODE = 'en'
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True
ROOMS = [dict(name ='room', display_name ='room', participant_label_file='participant_labels.txt')]
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = '1234'

DEMO_PAGE_INTRO_HTML = """ """

SECRET_KEY = '122'
