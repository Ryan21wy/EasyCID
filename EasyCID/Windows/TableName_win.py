# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Change_Name.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 87)
        self.name = QtWidgets.QLineEdit(Dialog)
        self.name.setGeometry(QtCore.QRect(120, 10, 271, 31))
        self.name.setObjectName("name")
        self.save_ = QtWidgets.QPushButton(Dialog)
        self.save_.setEnabled(True)
        self.save_.setGeometry(QtCore.QRect(150, 50, 93, 28))
        self.save_.setObjectName("save_")
        self.reset_ = QtWidgets.QPushButton(Dialog)
        self.reset_.setGeometry(QtCore.QRect(270, 50, 93, 28))
        self.reset_.setObjectName("reset_")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(0, 10, 121, 31))
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Modify table name"))
        self.save_.setText(_translate("Dialog", "save"))
        self.reset_.setText(_translate("Dialog", "reset"))
        self.label.setText(_translate("Dialog", "  Table Name："))