# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'training_parm.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_training_parm_win(object):
    def setupUi(self, training_parm_win):
        training_parm_win.setObjectName("training_parm_win")
        training_parm_win.resize(322, 286)
        self.reset_ = QtWidgets.QPushButton(training_parm_win)
        self.reset_.setGeometry(QtCore.QRect(180, 250, 93, 28))
        self.reset_.setObjectName("reset_")
        self.save_ = QtWidgets.QPushButton(training_parm_win)
        self.save_.setEnabled(False)
        self.save_.setGeometry(QtCore.QRect(40, 250, 93, 28))
        self.save_.setObjectName("save_")
        self.groupBox = QtWidgets.QGroupBox(training_parm_win)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 321, 181))
        self.groupBox.setObjectName("groupBox")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_5.setGeometry(QtCore.QRect(10, 20, 151, 31))
        self.textBrowser_5.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_5.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_5.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.optimizer_ = QtWidgets.QComboBox(self.groupBox)
        self.optimizer_.setGeometry(QtCore.QRect(160, 20, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.optimizer_.setFont(font)
        self.optimizer_.setObjectName("optimizer_")
        self.optimizer_.addItem("")
        self.optimizer_.addItem("")
        self.optimizer_.addItem("")
        self.optimizer_.addItem("")
        self.textBrowser_6 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_6.setGeometry(QtCore.QRect(10, 60, 151, 31))
        self.textBrowser_6.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_6.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_6.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_6.setObjectName("textBrowser_6")
        self.learnrate = QtWidgets.QLineEdit(self.groupBox)
        self.learnrate.setGeometry(QtCore.QRect(160, 60, 151, 31))
        self.learnrate.setStyleSheet("")
        self.learnrate.setObjectName("learnrate")
        self.textBrowser_7 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_7.setGeometry(QtCore.QRect(10, 100, 151, 31))
        self.textBrowser_7.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_7.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_7.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_7.setObjectName("textBrowser_7")
        self.batchsize = QtWidgets.QSpinBox(self.groupBox)
        self.batchsize.setGeometry(QtCore.QRect(160, 100, 151, 31))
        self.batchsize.setMaximum(99999)
        self.batchsize.setProperty("value", 512)
        self.batchsize.setObjectName("batchsize")
        self.textBrowser_8 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_8.setGeometry(QtCore.QRect(10, 140, 151, 31))
        self.textBrowser_8.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_8.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_8.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_8.setObjectName("textBrowser_8")
        self.epochs_ = QtWidgets.QSpinBox(self.groupBox)
        self.epochs_.setGeometry(QtCore.QRect(160, 140, 151, 31))
        self.epochs_.setMaximum(99999)
        self.epochs_.setProperty("value", 100)
        self.epochs_.setObjectName("epochs_")
        self.groupBox_2 = QtWidgets.QGroupBox(training_parm_win)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 180, 321, 61))
        self.groupBox_2.setObjectName("groupBox_2")
        self.textBrowser_9 = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_9.setGeometry(QtCore.QRect(10, 20, 151, 31))
        self.textBrowser_9.setStyleSheet("background-color: rgb(240, 240, 240);\n"
"border: soild 1px rgb(240, 240, 240);")
        self.textBrowser_9.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_9.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_9.setObjectName("textBrowser_9")
        self.noise = QtWidgets.QLineEdit(self.groupBox_2)
        self.noise.setGeometry(QtCore.QRect(160, 20, 151, 31))
        self.noise.setStyleSheet("")
        self.noise.setObjectName("noise")

        self.retranslateUi(training_parm_win)
        QtCore.QMetaObject.connectSlotsByName(training_parm_win)

    def retranslateUi(self, training_parm_win):
        _translate = QtCore.QCoreApplication.translate
        training_parm_win.setWindowTitle(_translate("training_parm_win", "Training Parameter"))
        self.reset_.setText(_translate("training_parm_win", "reset"))
        self.save_.setText(_translate("training_parm_win", "save"))
        self.groupBox.setTitle(_translate("training_parm_win", "Training"))
        self.textBrowser_5.setHtml(_translate("training_parm_win", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">optimizer</span></p></body></html>"))
        self.optimizer_.setItemText(0, _translate("training_parm_win", "Adam"))
        self.optimizer_.setItemText(1, _translate("training_parm_win", "Adadelta"))
        self.optimizer_.setItemText(2, _translate("training_parm_win", "Adagrad"))
        self.optimizer_.setItemText(3, _translate("training_parm_win", "Adamax"))
        self.textBrowser_6.setHtml(_translate("training_parm_win", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">learning rate</span></p></body></html>"))
        self.learnrate.setText(_translate("training_parm_win", "0.0001"))
        self.textBrowser_7.setHtml(_translate("training_parm_win", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">batch size</span></p></body></html>"))
        self.textBrowser_8.setHtml(_translate("training_parm_win", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">epochs</span></p></body></html>"))
        self.groupBox_2.setTitle(_translate("training_parm_win", "Noise"))
        self.textBrowser_9.setHtml(_translate("training_parm_win", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">rate</span></p></body></html>"))
        self.noise.setText(_translate("training_parm_win", "0.005"))
