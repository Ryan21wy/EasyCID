from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("dialog")
        dialog.resize(323, 145)
        self.OK = QtWidgets.QPushButton(dialog)
        self.OK.setEnabled(True)
        self.OK.setGeometry(QtCore.QRect(50, 100, 93, 28))
        self.OK.setObjectName("OK")
        self.label = QtWidgets.QLabel(dialog)
        self.label.setGeometry(QtCore.QRect(10, 10, 131, 31))
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(dialog)
        self.comboBox.setGeometry(QtCore.QRect(160, 10, 151, 31))
        self.comboBox.setObjectName("comboBox")
        self.label_2 = QtWidgets.QLabel(dialog)
        self.label_2.setGeometry(QtCore.QRect(10, 50, 131, 31))
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.Threshold = QtWidgets.QDoubleSpinBox(dialog)
        self.Threshold.setGeometry(QtCore.QRect(160, 50, 151, 31))
        self.Threshold.setFrame(False)
        self.Threshold.setPrefix("")
        self.Threshold.setMaximum(1.0)
        self.Threshold.setSingleStep(0.01)
        self.Threshold.setProperty("value", 0.5)
        self.Threshold.setObjectName("Threshold")
        self.cancel_ = QtWidgets.QPushButton(dialog)
        self.cancel_.setEnabled(True)
        self.cancel_.setGeometry(QtCore.QRect(180, 100, 93, 28))
        self.cancel_.setObjectName("cancel_")

        self.retranslateUi(dialog)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("dialog", "Prediction Setting"))
        self.OK.setText(_translate("dialog", "OK"))
        self.label.setText(_translate("dialog", "Model Selection:"))
        self.label_2.setText(_translate("dialog", "Threshold："))
        self.cancel_.setText(_translate("dialog", "cancel"))
