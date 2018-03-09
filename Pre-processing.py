## Talking Data Project
## Pre-processing steps

import numpy as np
import pandas as pd

# Pre-processing 1
# Translate phone brand name from Chinese into English.

# Dictionary of phone brands
english_phone_brands_mapping = {
    "三星": "samsung",
    "天语": "Ktouch",
    "海信": "hisense",
    "联想": "lenovo",
    "欧比": "obi",
    "爱派尔": "ipair",
    "努比亚": "nubia",
    "优米": "youmi",
    "朵唯": "dowe",
    "黑米": "heymi",
    "锤子": "hammer",
    "酷比魔方": "koobee",
    "美图": "meitu",
    "尼比鲁": "nibilu",
    "一加": "oneplus",
    "优购": "yougo",
    "诺基亚": "nokia",
    "糖葫芦": "candy",
    "中国移动": "ccmc",
    "语信": "yuxin",
    "基伍": "kiwu",
    "青橙": "greeno",
    "华硕": "asus",
    "夏新": "panosonic",
    "维图": "weitu",
    "艾优尼": "aiyouni",
    "摩托罗拉": "moto",
    "乡米": "xiangmi",
    "米奇": "micky",
    "大可乐": "bigcola",
    "沃普丰": "wpf",
    "神舟": "hasse",
    "摩乐": "mole",
    "飞秒": "fs",
    "米歌": "mige",
    "富可视": "fks",
    "德赛": "desci",
    "梦米": "mengmi",
    "乐视": "lshi",
    "小杨树": "smallt",
    "纽曼": "newman",
    "邦华": "banghua",
    "E派": "epai",
    "易派": "epai",
    "普耐尔": "pner",
    "欧新": "ouxin",
    "西米": "ximi",
    "海尔": "haier",
    "波导": "bodao",
    "糯米": "nuomi",
    "唯米": "weimi",
    "酷珀": "kupo",
    "谷歌": "google",
    "昂达": "ada",
    "聆韵": "lingyun",
    "小米": "Xiaomi",
    "华为": "Huawei",
    "魅族": "Meizu",
    "中兴": "ZTE",
    "酷派": "Coolpad",
    "金立": "Gionee",
    "SUGAR": "SUGAR",
    "OPPO": "OPPO",
    "vivo": "vivo",
    "HTC": "HTC",
    "LG": "LG",
    "ZUK": "ZUK",
    "TCL": "TCL",
    "LOGO": "LOGO",
    "SUGAR": "SUGAR",
    "Lovme": "Lovme",
    "PPTV": "PPTV",
    "ZOYE": "ZOYE",
    "MIL": "MIL",
    "索尼" : "Sony",
    "欧博信" : "Opssom",
    "奇酷" : "Qiku",
    "酷比" : "CUBE",
    "康佳" : "Konka",
    "亿通" : "Yitong",
    "金星数码" : "JXD",
    "至尊宝" : "Monkey King",
    "百立丰" : "Hundred Li Feng",
    "贝尔丰" : "Bifer",
    "百加" : "Bacardi",
    "诺亚信" : "Noain",
    "广信" : "Kingsun",
    "世纪天元" : "Ctyon",
    "青葱" : "Cong",
    "果米" : "Taobao",
    "斐讯" : "Phicomm",
    "长虹" : "Changhong",
    "欧奇" : "Oukimobile",
    "先锋" : "XFPLAY",
    "台电" : "Teclast",
    "大Q" : "Daq",
    "蓝魔" : "Ramos",
    "奥克斯" : "AUX",
    "飞利浦": "Philips",
    "智镁": "Zhimei",
    "惠普": "HP",
    "原点": "Origin",
    "戴尔": "Dell",
    "碟米": "Diemi",
    "西门子": "Siemens",
    "亚马逊": "Amazon",
    "宏碁": "Acer"
}

phone_brand_device_model = pd.read_csv('./Input/phone_brand_device_model.csv')
phone_brand_device_model.drop_duplicates("device_id", keep='first', inplace=True)
phone_brand_device_model.phone_brand = phone_brand_device_model.phone_brand.map(pd.Series(english_phone_brands_mapping), na_action='ignore')


# Pre-processing 2
# Remove events based on the latitude and longitude
events = pd.read_csv('./Input/events.csv')

events_pre = events[events.latitude < 54]
events_pre = events[events.latitude > 20]
events_pre = events[events.longitude < 135]
events_pre = events[events.longitude >73]
'''
events_pre_1 = events[events.latitude != 0]
events_pre_1 = events[events.latitude != 1]
events_pre_1 = events[events.longitude != 0]
events_pre_1 = events[events.longitude != 1]
'''


# Pre-processing 3
# Remove DUPLICATE record

gender_age_train = pd.read_csv('./Input/gender_age_train.csv')
gender_age_train.drop_duplicates(keep='first', inplace=True)


# Checking preprocessing
# Write to CSV
# phone_brand_device_model.to_csv('./Input/phone_brand_device_model_CHtoEN.csv', mode = 'w', header=True, index=False)
# gender_age_train.to_csv('./Input/gender_age_train_updated.csv', mode = 'w', header=True, index=False)
# events_pre.to_csv('./Input/events_updated.csv', mode = 'w', header=True, index=False)
# events_pre_1.to_csv('./Input/events_updated_1.csv', mode = 'w', header=True, index=False)
