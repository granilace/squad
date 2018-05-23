POS_SIZE = 50
ENTITY_SIZE = 19
BATCH_SIZE = 32
NUM_FEATURES = 4

RNN_DROPOUT_RATE = 0.4
EMBEDDING_DROPOUT_RATE = 0.4
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 128
WORDS_IN_VOCABULARY = 91187
MAX_TEXT_LEN = 767
MAX_QUESTION_LEN = 60
MAX_ANSW_LEN = 15
VALID_SAMPLES = 1000

RAW_DATA_PATH = 'data/RAW_DATA'
RAW_META_PATH = 'data/RAW_META'

MODELS_PATH = 'keras_models/'
LAST_MODEL_NAME = 'last_model'

EMB_PATH = 'data/WORD_EMBEDDINGS'
POS_PATH = 'data/ONEHOT_POS'
ENT_PATH = 'data/ONEHOT_ENT'

TRAIN_CONTEXT_PATH = 'data/TRAIN_CONTEXTS'
TRAIN_QUESTION_PATH = 'data/TRAIN_QUESTIONS'
DEV_CONTEXT_PATH = 'data/DEV_CONTEXTS'
DEV_QUESTION_PATH = 'data/DEV_QUESTIONS'
TRAIN_ANSWER_BIN_PATH = 'data/TRAIN_ANSW_BIN'
TRAIN_ANSWER_PAIRS_PATH = 'data/TRAIN_ANSW_PAIRS'
DEV_ANSWER_PAIRS_PATH = 'data/DEV_ANSW_PAIRS'
DO_PADDING = True

if DO_PADDING:
    ENTITY_SIZE = 19 + 1
if DO_PADDING:
    POS_SIZE = 50 + 1

INPUT_CONTEXT_LEN = EMBEDDING_SIZE + NUM_FEATURES + POS_SIZE + ENTITY_SIZE
INPUT_QUESTION_LEN = EMBEDDING_SIZE
