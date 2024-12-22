DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'

DEFAULT_CLIENT_SERVICE_URLS = (
    'translate.googleapis.com',
)

DEFAULT_SERVICE_URLS = ('translate.google.ac', 'translate.google.ad', 'translate.google.ae',
                        'translate.google.al', 'translate.google.am', 'translate.google.as',
                        'translate.google.at', 'translate.google.az', 'translate.google.ba',
                        'translate.google.be', 'translate.google.bf', 'translate.google.bg',
                        'translate.google.bi', 'translate.google.bj', 'translate.google.bs',
                        'translate.google.bt', 'translate.google.by', 'translate.google.ca',
                        'translate.google.cat', 'translate.google.cc', 'translate.google.cd',
                        'translate.google.cf', 'translate.google.cg', 'translate.google.ch',
                        'translate.google.ci', 'translate.google.cl', 'translate.google.cm',
                        'translate.google.cn', 'translate.google.co.ao', 'translate.google.co.bw',
                        'translate.google.co.ck', 'translate.google.co.cr', 'translate.google.co.id',
                        'translate.google.co.il', 'translate.google.co.in', 'translate.google.co.jp',
                        'translate.google.co.ke', 'translate.google.co.kr', 'translate.google.co.ls',
                        'translate.google.co.ma', 'translate.google.co.mz', 'translate.google.co.nz',
                        'translate.google.co.th', 'translate.google.co.tz', 'translate.google.co.ug',
                        'translate.google.co.uk', 'translate.google.co.uz', 'translate.google.co.ve',
                        'translate.google.co.vi', 'translate.google.co.za', 'translate.google.co.zm',
                        'translate.google.co.zw', 'translate.google.com.af', 'translate.google.com.ag',
                        'translate.google.com.ai', 'translate.google.com.ar', 'translate.google.com.au',
                        'translate.google.com.bd', 'translate.google.com.bh', 'translate.google.com.bn',
                        'translate.google.com.bo', 'translate.google.com.br', 'translate.google.com.bz',
                        'translate.google.com.co', 'translate.google.com.cu', 'translate.google.com.cy',
                        'translate.google.com.do', 'translate.google.com.ec', 'translate.google.com.eg',
                        'translate.google.com.et', 'translate.google.com.fj', 'translate.google.com.gh',
                        'translate.google.com.gi', 'translate.google.com.gt', 'translate.google.com.hk',
                        'translate.google.com.jm', 'translate.google.com.kh', 'translate.google.com.kw',
                        'translate.google.com.lb', 'translate.google.com.ly', 'translate.google.com.mm',
                        'translate.google.com.mt', 'translate.google.com.mx', 'translate.google.com.my',
                        'translate.google.com.na', 'translate.google.com.ng', 'translate.google.com.ni',
                        'translate.google.com.np', 'translate.google.com.om', 'translate.google.com.pa',
                        'translate.google.com.pe', 'translate.google.com.pg', 'translate.google.com.ph',
                        'translate.google.com.pk', 'translate.google.com.pr', 'translate.google.com.py',
                        'translate.google.com.qa', 'translate.google.com.sa', 'translate.google.com.sb',
                        'translate.google.com.sg', 'translate.google.com.sl', 'translate.google.com.sv',
                        'translate.google.com.tj', 'translate.google.com.tr', 'translate.google.com.tw',
                        'translate.google.com.ua', 'translate.google.com.uy', 'translate.google.com.vc',
                        'translate.google.com.vn', 'translate.google.com', 'translate.google.cv',
                        'translate.google.cz', 'translate.google.de', 'translate.google.dj',
                        'translate.google.dk', 'translate.google.dm', 'translate.google.dz',
                        'translate.google.ee', 'translate.google.es', 'translate.google.eu',
                        'translate.google.fi', 'translate.google.fm', 'translate.google.fr',
                        'translate.google.ga', 'translate.google.ge', 'translate.google.gf',
                        'translate.google.gg', 'translate.google.gl', 'translate.google.gm',
                        'translate.google.gp', 'translate.google.gr', 'translate.google.gy',
                        'translate.google.hn', 'translate.google.hr', 'translate.google.ht',
                        'translate.google.hu', 'translate.google.ie', 'translate.google.im',
                        'translate.google.io', 'translate.google.iq', 'translate.google.is',
                        'translate.google.it', 'translate.google.je', 'translate.google.jo',
                        'translate.google.kg', 'translate.google.ki', 'translate.google.kz',
                        'translate.google.la', 'translate.google.li', 'translate.google.lk',
                        'translate.google.lt', 'translate.google.lu', 'translate.google.lv',
                        'translate.google.md', 'translate.google.me', 'translate.google.mg',
                        'translate.google.mk', 'translate.google.ml', 'translate.google.mn',
                        'translate.google.ms', 'translate.google.mu', 'translate.google.mv',
                        'translate.google.mw', 'translate.google.ne', 'translate.google.nf',
                        'translate.google.nl', 'translate.google.no', 'translate.google.nr',
                        'translate.google.nu', 'translate.google.pl', 'translate.google.pn',
                        'translate.google.ps', 'translate.google.pt', 'translate.google.ro',
                        'translate.google.rs', 'translate.google.ru', 'translate.google.rw',
                        'translate.google.sc', 'translate.google.se', 'translate.google.sh',
                        'translate.google.si', 'translate.google.sk', 'translate.google.sm',
                        'translate.google.sn', 'translate.google.so', 'translate.google.sr',
                        'translate.google.st', 'translate.google.td', 'translate.google.tg',
                        'translate.google.tk', 'translate.google.tl', 'translate.google.tm',
                        'translate.google.tn', 'translate.google.to', 'translate.google.tt',
                        'translate.google.us', 'translate.google.vg', 'translate.google.vu',
                        'translate.google.ws')

SPECIAL_CASES = {
    'ee': 'et',
}

LANGUAGES = {
    "abk": "abkhaz",
    "ace": "acehnese",
    "ach": "acholi",
    "aar": "afar",
    "af": "afrikaans",
    "sq": "albanian",
    "alz": "alur",
    "am": "amharic",
    "ar": "arabic",
    "hy": "armenian",
    "as": "assamese",
    "ava": "avar",
    "awa": "awadhi",
    "ay": "aymara",
    "az": "azerbaijani",
    "ban": "balinese",
    "bal": "baluchi",
    "bm": "bambara",
    "bci": "baoulé",
    "bak": "bashkir",
    "eu": "basque",
    "btx": "batak karo",
    "bts": "batak simalungun",
    "bbc": "batak toba",
    "be": "belarusian",
    "bem": "bemba",
    "bn": "bengali",
    "bew": "betawi",
    "bho": "bhojpuri",
    "bik": "bikol",
    "bs": "bosnian",
    "bre": "breton",
    "bg": "bulgarian",
    "bua": "buryat",
    "yue": "cantonese",
    "ca": "catalan",
    "ceb": "cebuano",
    "cha": "chamorro",
    "che": "chechen",
    "zh": "chinese",
    "zh-cn": "chinese (simplified)",
    "zh-tw": "chinese (traditional)",
    "chk": "chuukese",
    "chv": "chuvash",
    "co": "corsican",
    "crh": "crimean tatar",
    "hr": "croatian",
    "cs": "czech",
    "da": "danish",
    "fa-af": "dari",
    "dv": "dhivehi",
    "din": "dinka",
    "doi": "dogri",
    "dom": "dombe",
    "nl": "dutch",
    "dyu": "dyula",
    "dzo": "dzongkha",
    "en": "english",
    "eo": "esperanto",
    "et": "estonian",
    "fao": "faroese",
    "fij": "fijian",
    "fil": "filipino (tagalog)",
    "fi": "finnish",
    "fon": "fon",
    "fr": "french",
    "fy": "frisian",
    "fur": "friulian",
    "ful": "fulani",
    "gaa": "ga",
    "gl": "galician",
    "ka": "georgian",
    "de": "german",
    "el": "greek",
    "gn": "guarani",
    "gu": "gujarati",
    "ht": "haitian creole",
    "cnh": "hakha chin",
    "ha": "hausa",
    "haw": "hawaiian",
    "he": "hebrew",
    "iw": "hebrew",
    "hil": "hiligaynon",
    "hi": "hindi",
    "hmn": "hmong",
    "hu": "hungarian",
    "hrx": "hunsrik",
    "iba": "iban",
    "is": "icelandic",
    "ig": "igbo",
    "ilo": "ilocano",
    "id": "indonesian",
    "ga": "irish",
    "it": "italian",
    "jam": "jamaican patois",
    "ja": "japanese",
    "jv": "javanese",
    "jw": "javanese",
    "kac": "jingpo",
    "kal": "kalaallisut",
    "kn": "kannada",
    "kau": "kanuri",
    "pam": "kapampangan",
    "kk": "kazakh",
    "kha": "khasi",
    "km": "khmer",
    "cgg": "kiga",
    "kik": "kikongo",
    "rw": "kinyarwanda",
    "ktu": "kituba",
    "trp": "kokborok",
    "kom": "komi",
    "gom": "konkani",
    "ko": "korean",
    "kri": "krio",
    "ku": "kurdish",
    "ckb": "kurdish (sorani)",
    "ky": "kyrgyz",
    "lo": "lao",
    "ltg": "latgalian",
    "la": "latin",
    "lv": "latvian",
    "lij": "ligurian",
    "lim": "limburgish",
    "ln": "lingala",
    "lt": "lithuanian",
    "lmo": "lombard",
    "lg": "luganda",
    "luo": "luo",
    "lb": "luxembourgish",
    "mk": "macedonian",
    "mad": "madurese",
    "mai": "maithili",
    "mak": "makassar",
    "mg": "malagasy",
    "ms": "malay",
    "ms-arab": "malay (jawi)",
    "ml": "malayalam",
    "mt": "maltese",
    "mam": "mam",
    "glv": "manx",
    "mi": "maori",
    "mr": "marathi",
    "mah": "marshallese",
    "mwr": "marwadi",
    "mfe": "mauritian creole",
    "mhr": "meadow mari",
    "mni-mtei": "meiteilon (manipuri)",
    "min": "minang",
    "lus": "mizo",
    "mn": "mongolian",
    "my": "myanmar (burmese)",
    "nhe": "nahuatl (eastern huasteca)",
    "ndc-zw": "ndau",
    "nde": "ndebele (south)",
    "new": "nepalbhasa (newari)",
    "ne": "nepali",
    #'bm-nkoo': 'nko',
    "no": "norwegian",
    "nus": "nuer",
    "ny": "nyanja (chichewa)",
    "oci": "occitan",
    "or": "odia (oriya)",
    "om": "oromo",
    "oss": "ossetian",
    "pag": "pangasinan",
    "pap": "papiamento",
    "ps": "pashto",
    "fa": "persian",
    "pl": "polish",
    "por": "portuguese (portugal)",
    "pt": "portuguese (portugal, brazil)",
    "pa": "punjabi",
    "pa-arab": "punjabi (shahmukhi)",
    "kek": "q'eqchi'",
    "qu": "quechua",
    "rom": "romani",
    "ro": "romanian",
    "run": "rundi",
    "ru": "russian",
    "sme": "sami (north)",
    "sm": "samoan",
    "sag": "sango",
    "sa": "sanskrit",
    "sat": "santali",
    "gd": "scots gaelic",
    "nso": "sepedi",
    "sr": "serbian",
    "st": "sesotho",
    "crs": "seychellois creole",
    "shn": "shan",
    "sn": "shona",
    "scn": "sicilian",
    "szl": "silesian",
    "sd": "sindhi",
    "si": "sinhala (sinhalese)",
    "sk": "slovak",
    "sl": "slovenian",
    "so": "somali",
    "es": "spanish",
    "su": "sundanese",
    "sus": "susu",
    "sw": "swahili",
    "ssw": "swati",
    "sv": "swedish",
    "tl": "tagalog (filipino)",
    "tah": "tahitian",
    "tg": "tajik",
    "ber-atn": "tamazight",
    "ber": "tamazight (tifinagh)",
    "ta": "tamil",
    "tt": "tatar",
    "te": "telugu",
    "tet": "tetum",
    "th": "thai",
    "bod": "tibetan",
    "ti": "tigrinya",
    "tiv": "tiv",
    "tpi": "tok pisin",
    "ton": "tongan",
    "ts": "tsonga",
    "tsn": "tswana",
    "tcy": "tulu",
    "tum": "tumbuka",
    "tr": "turkish",
    "tk": "turkmen",
    "tuk": "tuvan",
    "ak": "twi (akan)",
    "udm": "udmurt",
    "uk": "ukrainian",
    "ur": "urdu",
    "ug": "uyghur",
    "uz": "uzbek",
    "ven": "venda",
    "vec": "venetian",
    "vi": "vietnamese",
    "war": "waray",
    "cy": "welsh",
    "wol": "wolof",
    "xh": "xhosa",
    "sah": "yakut",
    "yi": "yiddish",
    "yo": "yoruba",
    "yua": "yucatec maya",
    "zap": "zapotec",
    "zu": "zulu",
}

LANGCODES = dict(map(reversed, LANGUAGES.items()))
DEFAULT_RAISE_EXCEPTION = False
DUMMY_DATA = [[["", None, None, 0]], None, "en", None,
              None, None, 1, None, [["en"], None, [1], ["en"]]]