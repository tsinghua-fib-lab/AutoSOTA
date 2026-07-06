import json
import os.path as osp
import os
import pickle
import shutil
from database_utils import query_all_stocks, Database_operate


def load_persona(json_path):
    with open(json_path, 'r') as file:
        persona = json.load(file)
       # print(persona)

    return persona


def load_stocks(json_path):
    with open(json_path, 'r') as file:
        stocks = json.load(file)
       # print(stocks)

    return stocks


def save_class(name, obj, Save_Path):
    class_save_path = osp.join(Save_Path, "classes")
    if not os.path.exists(class_save_path):
        os.makedirs(class_save_path)
    obj.db = None
    save_file = osp.join(class_save_path, name+".pkl")
    output = open(save_file, "wb")
    obj_pkl = pickle.dumps(obj)
    output.write(obj_pkl)
    output.close()


def load_class(name, args):
    class_save_path = osp.join(args.Save_Path, "classes")
    save_file = osp.join(class_save_path, name+".pkl")
    with open(save_file, "rb") as file:
        obj = pickle.loads(file.read())
    return obj


def save_all(virtual_date, iteration, stocks, market_index, persons, market, args):
    infor_dic = {"virtual_date": virtual_date, "iteration": iteration}
    infor_json = json.dumps(infor_dic, indent=4)
    with open(osp.join(args.Save_Path, "information.json"), "w") as file:
        file.write(infor_json)
    database = market.db

    for index, each in enumerate(stocks):
        each.db = None
        save_name = "STOCK_{}".format(index)
        save_class(save_name, each, args.Save_Path)

    persons[-1].db = None
    for index, each in enumerate(persons):
        each.db = None
        save_name = "PERSON_{}".format(index) # including the broker
        save_class(save_name, each)

    market_index.db = None
    save_name = "Market_index"
    save_class(save_name, market_index)

    market.db = None
    save_name = "MARKET"
    save_class(save_name, market)

    for index, each in enumerate(stocks):
        each.db = database
    for index, each in enumerate(persons):
        each.db = database
    market_index.db = database
    market.db = database




def load_all(args):
    database = Database_operate(osp.join(args.Save_Path, "data"))
    with open(osp.join(args.Save_Path, "information.json"), 'r') as file:
        infor_dic = json.load(file)
    current_date = infor_dic["virtual_date"]
    current_iteration = infor_dic["iteration"]#

    stocks = []
    for index in range(args.Num_Stock):
        save_name = "STOCK_{}".format(index)
        obj = load_class(save_name, args)
        obj.db = database
        stocks.append(obj)

    save_name = "PERSON_{}".format(args.Num_Person)
    broker = load_class(save_name)
    broker.stocks = stocks
    broker.db = database

    persons = []
    for index in range(args.Num_Person):
        save_name = "PERSON_{}".format(index)
        obj = load_class(save_name)
        obj.db = database
        obj.stocks = stocks
        obj.broker = broker
        persons.append(obj)

    persons.append(broker)

    save_name = "Market_index"
    market_index = load_class(save_name)
    market_index.db = database
    market_index.stocks = stocks

    save_name = "MARKET"
    market = load_class(save_name)
    market.db = database
    market.broker = broker
    market.stocks = stocks
    market.persons = persons
    return current_date, current_iteration, broker, market_index, market, stocks, persons








if __name__ == "__main__":
    load_persona()
    load_stocks()
