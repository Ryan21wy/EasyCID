# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ModelSelect_win.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("dialog")
        dialog.resize(323, 95)
        self.OK = QtWidgets.QPushButton(dialog)
        self.OK.setEnabled(True)
        self.OK.setGeometry(QtCore.QRect(110, 60, 93, 28))
        self.OK.setObjectName("OK")
        self.label = QtWidgets.QLabel(dialog)
        self.label.setGeometry(QtCore.QRect(10, 10, 141, 31))
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(dialog)
        self.comboBox.setGeometry(QtCore.QRect(160, 10, 151, 31))
        self.comboBox.setObjectName("comboBox")

        self.retranslateUi(dialog)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("dialog", "Prediction"))
        self.OK.setText(_translate("dialog", "OK"))
        self.label.setText(_translate("dialog", "Model Selection:"))
