import sqlite3


class EasyCIDDatabase:
    def __init__(self, path):
        super(EasyCIDDatabase, self).__init__()
        self.db = sqlite3.connect(path)
        self.cur = self.db.cursor()
        self.db.execute('pragma foreign_keys=on')

    def set_up_database(self):
        sql = 'CREATE TABLE Groups (Group_ID INTEGER PRIMARY KEY, Group_Name VARCHAR)'
        self.cur.execute(sql)
        sql = 'CREATE TABLE Component_Info (Component_ID INTEGER PRIMARY KEY, Component_Name VARCHAR, ' \
              'Raw_Spectrum BLOB, Raw_Axis BLOB, Inter_Time FLOAT, Model INTEGER, From_Group INTEGER, ' \
              'foreign key(From_Group) references Groups(Group_ID) on delete cascade on update cascade) '
        self.cur.execute(sql)
        sql = 'CREATE TABLE Group_Model_Info (Raman_Start FLOAT, Raman_End FLOAT, Raman_Interval FLOAT, ' \
              'Aug_Save_Path VARCHAR, Save_Path VARCHAR, From_Group INTEGER, foreign key(From_Group) references ' \
              'Groups(Group_ID) on delete cascade on update cascade) '
        self.cur.execute(sql)
        sql = 'CREATE TABLE Component_Model_Info (Augment_Num INTEGER, Noise_Rate FLOAT, Optimizer INTEGER, ' \
              'LR FLOAT, BS INTEGER, EPS INTEGER, From_Component INTEGER, foreign key(From_Component) references ' \
              'Component_Info(Component_ID) on delete cascade on update cascade) '
        self.cur.execute(sql)
        self.db.commit()

    def select(self, query, table, constrain=None, constrain_value=None):
        if constrain:
            sql = 'select ' + query + ' from ' + table + ' where ' + constrain
            self.cur.execute(sql, constrain_value)
        else:
            sql = 'select ' + query + ' from ' + table
            self.cur.execute(sql)
        data = self.cur.fetchall()
        return data

    def insert(self, table, values, data, many=False):
        sql = 'insert into ' + table + ' VALUES ' + values
        if many:
            self.cur.executemany(sql, data)
        else:
            self.cur.execute(sql, data)
        self.db.commit()

    def delete(self, table, constrain, constrain_value):
        sql = 'delete from ' + table + ' where ' + constrain
        self.cur.execute(sql, constrain_value)
        self.db.commit()

    def update(self, table, query, constrain, data):
        sql = 'update ' + table + ' set ' + query + ' where ' + constrain
        self.cur.execute(sql, data)
        self.db.commit()