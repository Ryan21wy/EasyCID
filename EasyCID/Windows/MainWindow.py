# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'EasyCID_demo.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1229, 812)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setMinimumSize(QtCore.QSize(0, 35))
        self.frame_3.setMaximumSize(QtCore.QSize(16777215, 35))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.frame_3.setStyleSheet("QFrame{background-color:rgb(50, 100, 170)}"
                                   "QPushButton{background-color:rgb(50, 100, 170);"
                                   "color:rgb(255, 255, 255);"
                                   "border:0px;}"
                                   "QPushButton:hover{background-color:rgb(30, 75, 125)}"
                                   )
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.data_ = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.data_.sizePolicy().hasHeightForWidth())
        self.data_.setSizePolicy(sizePolicy)
        self.data_.setMinimumSize(QtCore.QSize(100, 35))
        self.data_.setObjectName("data_")
        self.horizontalLayout.addWidget(self.data_)
        self.function_ = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.function_.sizePolicy().hasHeightForWidth())
        self.function_.setSizePolicy(sizePolicy)
        self.function_.setMinimumSize(QtCore.QSize(100, 35))
        self.function_.setObjectName("function_")
        self.horizontalLayout.addWidget(self.function_)
        self.help_ = QtWidgets.QPushButton(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.help_.sizePolicy().hasHeightForWidth())
        self.help_.setSizePolicy(sizePolicy)
        self.help_.setMinimumSize(QtCore.QSize(100, 35))
        self.help_.setObjectName("help_")
        self.horizontalLayout.addWidget(self.help_)
        self.database_path = QtWidgets.QLabel(self.frame_3)
        self.database_path.setObjectName("database_path")
        self.database_path.setStyleSheet("color:rgb(255, 255, 255)")
        self.horizontalLayout.addWidget(self.database_path)
        self.verticalLayout.addWidget(self.frame_3)
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setMinimumSize(QtCore.QSize(0, 80))
        self.stackedWidget.setMaximumSize(QtCore.QSize(16777215, 80))
        self.stackedWidget.setObjectName("stackedWidget")
        self.stackedWidget.setStyleSheet("QStackedWidget{background-color:rgb(240, 240, 240)}"
                                         "QToolButton{background-color:rgb(240, 240, 240);"
                                         "border:0px;}"
                                         "QToolButton:hover{background-color:rgb(220, 220, 220)}"
                                         "QFrame{background-color:rgb(240, 240, 240);"
                                         "border-right:1px solid rgb(200, 200, 200);}"
                                         "QLabel{background-color:rgb(240, 240, 240);"
                                         "border:0px;}"
                                         )
        self.data_page = QtWidgets.QWidget()
        self.data_page.setObjectName("data_page")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.data_page)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setSpacing(0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.frame_4 = QtWidgets.QFrame(self.data_page)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.open_database = QtWidgets.QToolButton(self.frame_4)
        self.open_database.setMinimumSize(QtCore.QSize(60, 60))
        self.open_database.setMaximumSize(QtCore.QSize(60, 60))
        self.open_database.setIconSize(QtCore.QSize(30, 30))
        self.open_database.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.open_database.setObjectName("open_database")
        self.horizontalLayout_7.addWidget(self.open_database)
        self.build_database = QtWidgets.QToolButton(self.frame_4)
        self.build_database.setMinimumSize(QtCore.QSize(60, 60))
        self.build_database.setMaximumSize(QtCore.QSize(60, 60))
        self.build_database.setBaseSize(QtCore.QSize(60, 60))
        self.build_database.setIconSize(QtCore.QSize(30, 30))
        self.build_database.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.build_database.setObjectName("build_database")
        self.horizontalLayout_7.addWidget(self.build_database)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.label = QtWidgets.QLabel(self.frame_4)
        self.label.setEnabled(False)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.horizontalLayout_10.addWidget(self.frame_4)
        self.frame_5 = QtWidgets.QFrame(self.data_page)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.add_group = QtWidgets.QToolButton(self.frame_5)
        self.add_group.setMinimumSize(QtCore.QSize(60, 60))
        self.add_group.setMaximumSize(QtCore.QSize(60, 60))
        self.add_group.setSizeIncrement(QtCore.QSize(0, 0))
        self.add_group.setBaseSize(QtCore.QSize(0, 0))
        self.add_group.setIconSize(QtCore.QSize(30, 30))
        self.add_group.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.add_group.setObjectName("add_group")
        self.horizontalLayout_8.addWidget(self.add_group)
        self.delete_group = QtWidgets.QToolButton(self.frame_5)
        self.delete_group.setMinimumSize(QtCore.QSize(60, 60))
        self.delete_group.setMaximumSize(QtCore.QSize(60, 60))
        self.delete_group.setSizeIncrement(QtCore.QSize(60, 60))
        self.delete_group.setIconSize(QtCore.QSize(30, 30))
        self.delete_group.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.delete_group.setObjectName("delete_group")
        self.horizontalLayout_8.addWidget(self.delete_group)
        self.verticalLayout_4.addLayout(self.horizontalLayout_8)
        self.label_2 = QtWidgets.QLabel(self.frame_5)
        self.label_2.setEnabled(False)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2)
        self.horizontalLayout_10.addWidget(self.frame_5)
        self.frame_6 = QtWidgets.QFrame(self.data_page)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.add_spectra = QtWidgets.QToolButton(self.frame_6)
        self.add_spectra.setMinimumSize(QtCore.QSize(60, 60))
        self.add_spectra.setSizeIncrement(QtCore.QSize(60, 60))
        self.add_spectra.setIconSize(QtCore.QSize(30, 30))
        self.add_spectra.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.add_spectra.setObjectName("add_spectra")
        self.horizontalLayout_9.addWidget(self.add_spectra)
        self.delete_spectra = QtWidgets.QToolButton(self.frame_6)
        self.delete_spectra.setMinimumSize(QtCore.QSize(60, 60))
        self.delete_spectra.setMaximumSize(QtCore.QSize(60, 60))
        self.delete_spectra.setSizeIncrement(QtCore.QSize(60, 60))
        self.delete_spectra.setIconSize(QtCore.QSize(30, 30))
        self.delete_spectra.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.delete_spectra.setObjectName("delete_spectra")
        self.horizontalLayout_9.addWidget(self.delete_spectra)
        self.verticalLayout_5.addLayout(self.horizontalLayout_9)
        self.label_3 = QtWidgets.QLabel(self.frame_6)
        self.label_3.setEnabled(False)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_5.addWidget(self.label_3)
        self.horizontalLayout_10.addWidget(self.frame_6)
        spacerItem = QtWidgets.QSpacerItem(792, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem)
        self.stackedWidget.addWidget(self.data_page)
        self.function_page = QtWidgets.QWidget()
        self.function_page.setObjectName("function_page")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.function_page)
        self.horizontalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_13.setSpacing(0)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.frame_10 = QtWidgets.QFrame(self.function_page)
        self.frame_10.setMinimumSize(QtCore.QSize(126, 80))
        self.frame_10.setMaximumSize(QtCore.QSize(126, 80))
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.frame_10)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setSpacing(0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.clear_plot_area = QtWidgets.QToolButton(self.frame_10)
        self.clear_plot_area.setMinimumSize(QtCore.QSize(60, 60))
        self.clear_plot_area.setSizeIncrement(QtCore.QSize(60, 60))
        self.clear_plot_area.setIconSize(QtCore.QSize(30, 30))
        self.clear_plot_area.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.clear_plot_area.setObjectName("clear_plot_area")
        self.horizontalLayout_16.addWidget(self.clear_plot_area)
        self.collect_spectra = QtWidgets.QToolButton(self.frame_10)
        self.collect_spectra.setMinimumSize(QtCore.QSize(60, 60))
        self.collect_spectra.setMaximumSize(QtCore.QSize(60, 60))
        self.collect_spectra.setSizeIncrement(QtCore.QSize(60, 60))
        self.collect_spectra.setIconSize(QtCore.QSize(30, 30))
        self.collect_spectra.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.collect_spectra.setObjectName("collect_spectra")
        self.horizontalLayout_16.addWidget(self.collect_spectra)
        self.verticalLayout_9.addLayout(self.horizontalLayout_16)
        self.label_12 = QtWidgets.QLabel(self.frame_10)
        self.label_12.setEnabled(False)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_9.addWidget(self.label_12)
        self.horizontalLayout_13.addWidget(self.frame_10)
        self.frame_7 = QtWidgets.QFrame(self.function_page)
        self.frame_7.setMinimumSize(QtCore.QSize(80, 80))
        self.frame_7.setMaximumSize(QtCore.QSize(80, 80))
        self.frame_7.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setSpacing(0)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem1)
        self.train_run = QtWidgets.QToolButton(self.frame_7)
        self.train_run.setMinimumSize(QtCore.QSize(60, 60))
        self.train_run.setMaximumSize(QtCore.QSize(60, 60))
        self.train_run.setSizeIncrement(QtCore.QSize(60, 60))
        self.train_run.setIconSize(QtCore.QSize(30, 30))
        self.train_run.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.train_run.setObjectName("train_run")
        self.horizontalLayout_11.addWidget(self.train_run)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem2)
        self.verticalLayout_6.addLayout(self.horizontalLayout_11)
        self.label_4 = QtWidgets.QLabel(self.frame_7)
        self.label_4.setEnabled(False)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_6.addWidget(self.label_4)
        self.horizontalLayout_13.addWidget(self.frame_7)
        self.frame_9 = QtWidgets.QFrame(self.function_page)
        self.frame_9.setMinimumSize(QtCore.QSize(250, 80))
        self.frame_9.setMaximumSize(QtCore.QSize(250, 80))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setSpacing(0)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.open_mix = QtWidgets.QToolButton(self.frame_9)
        self.open_mix.setMinimumSize(QtCore.QSize(60, 60))
        self.open_mix.setSizeIncrement(QtCore.QSize(60, 60))
        self.open_mix.setIconSize(QtCore.QSize(30, 30))
        self.open_mix.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.open_mix.setObjectName("open_mix")
        self.horizontalLayout_12.addWidget(self.open_mix)
        self.pred_run = QtWidgets.QToolButton(self.frame_9)
        self.pred_run.setMinimumSize(QtCore.QSize(60, 60))
        self.pred_run.setMaximumSize(QtCore.QSize(60, 60))
        self.pred_run.setSizeIncrement(QtCore.QSize(60, 60))
        self.pred_run.setIconSize(QtCore.QSize(30, 30))
        self.pred_run.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.pred_run.setObjectName("pred_run")
        self.horizontalLayout_12.addWidget(self.pred_run)
        self.ratio_estimation = QtWidgets.QToolButton(self.frame_9)
        self.ratio_estimation.setMinimumSize(QtCore.QSize(60, 60))
        self.ratio_estimation.setMaximumSize(QtCore.QSize(60, 60))
        self.ratio_estimation.setSizeIncrement(QtCore.QSize(60, 60))
        self.ratio_estimation.setIconSize(QtCore.QSize(30, 30))
        self.ratio_estimation.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.ratio_estimation.setObjectName("ratio_estimation")
        self.horizontalLayout_12.addWidget(self.ratio_estimation)
        self.save_results = QtWidgets.QToolButton(self.frame_9)
        self.save_results.setMinimumSize(QtCore.QSize(60, 60))
        self.save_results.setMaximumSize(QtCore.QSize(60, 60))
        self.save_results.setSizeIncrement(QtCore.QSize(60, 60))
        self.save_results.setIconSize(QtCore.QSize(30, 30))
        self.save_results.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.save_results.setObjectName("save_results")
        self.horizontalLayout_12.addWidget(self.save_results)
        self.verticalLayout_7.addLayout(self.horizontalLayout_12)
        self.label_11 = QtWidgets.QLabel(self.frame_9)
        self.label_11.setEnabled(False)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_7.addWidget(self.label_11)
        self.horizontalLayout_13.addWidget(self.frame_9)
        spacerItem3 = QtWidgets.QSpacerItem(830, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_13.addItem(spacerItem3)
        self.stackedWidget.addWidget(self.function_page)
        self.help_page = QtWidgets.QWidget()
        self.help_page.setObjectName("help_page")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.help_page)
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_14.setSpacing(0)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.frame_8 = QtWidgets.QFrame(self.help_page)
        self.frame_8.setMinimumSize(QtCore.QSize(126, 80))
        self.frame_8.setMaximumSize(QtCore.QSize(126, 80))
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame_8)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setSpacing(0)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.show_help_html = QtWidgets.QToolButton(self.frame_8)
        self.show_help_html.setMinimumSize(QtCore.QSize(60, 60))
        self.show_help_html.setSizeIncrement(QtCore.QSize(60, 60))
        self.show_help_html.setIconSize(QtCore.QSize(30, 30))
        self.show_help_html.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.show_help_html.setObjectName("show_help_html")
        self.horizontalLayout_15.addWidget(self.show_help_html)
        self.show_demo = QtWidgets.QToolButton(self.frame_8)
        self.show_demo.setMinimumSize(QtCore.QSize(60, 60))
        self.show_demo.setMaximumSize(QtCore.QSize(60, 60))
        self.show_demo.setSizeIncrement(QtCore.QSize(60, 60))
        self.show_demo.setIconSize(QtCore.QSize(30, 30))
        self.show_demo.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.show_demo.setObjectName("show_demo")
        self.horizontalLayout_15.addWidget(self.show_demo)
        self.verticalLayout_8.addLayout(self.horizontalLayout_15)
        self.label_5 = QtWidgets.QLabel(self.frame_8)
        self.label_5.setEnabled(False)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_8.addWidget(self.label_5)
        self.horizontalLayout_14.addWidget(self.frame_8)
        spacerItem4 = QtWidgets.QSpacerItem(1098, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem4)
        self.stackedWidget.addWidget(self.help_page)
        self.verticalLayout.addWidget(self.stackedWidget)
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setOpaqueResize(True)
        self.splitter.setHandleWidth(0)
        self.splitter.setObjectName("splitter")
        self.tabWidget = QtWidgets.QTabWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(350, 0))
        self.tabWidget.setMaximumSize(QtCore.QSize(350, 16777215))
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.West)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.data_display = QtWidgets.QTreeWidget(self.tab_2)
        self.data_display.setColumnCount(2)
        self.data_display.setObjectName("data_display")
        self.data_display.headerItem().setText(0, "1")
        self.data_display.headerItem().setText(1, "2")
        self.data_display.header().setHighlightSections(False)
        self.data_display.header().setSortIndicatorShown(True)
        self.horizontalLayout_4.addWidget(self.data_display)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.tab)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.data_list = QtWidgets.QListView(self.tab)
        self.data_list.setObjectName("data_list")
        self.horizontalLayout_5.addWidget(self.data_list)
        self.tabWidget.addTab(self.tab, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.tab_3)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.predict_result = QtWidgets.QTreeWidget(self.tab_3)
        self.predict_result.setObjectName("predict_result")
        self.predict_result.headerItem().setText(0, "1")
        self.horizontalLayout_6.addWidget(self.predict_result)
        self.tabWidget.addTab(self.tab_3, "")
        self.frame = QtWidgets.QFrame(self.splitter)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout.addWidget(self.splitter)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.frame_2.setStyleSheet("border:1px solid rgb(205, 205, 205);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser_3.sizePolicy().hasHeightForWidth())
        self.textBrowser_3.setSizePolicy(sizePolicy)
        self.textBrowser_3.setMinimumSize(QtCore.QSize(0, 25))
        self.textBrowser_3.setMaximumSize(QtCore.QSize(1000, 25))
        self.textBrowser_3.setStyleSheet("border:1px solid rgb(218, 218, 218);\n"
"background-color: rgb(218, 218, 218);")
        self.textBrowser_3.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.horizontalLayout_2.addWidget(self.textBrowser_3)
        self.progressBar = QtWidgets.QProgressBar(self.frame_2)
        self.progressBar.setMinimumSize(QtCore.QSize(0, 25))
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 25))
        self.progressBar.setStyleSheet("background-color: rgb(218, 218, 218);")
        self.progressBar.setProperty("value", 24)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_2.addWidget(self.progressBar)
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser_4.sizePolicy().hasHeightForWidth())
        self.textBrowser_4.setSizePolicy(sizePolicy)
        self.textBrowser_4.setMinimumSize(QtCore.QSize(0, 25))
        self.textBrowser_4.setMaximumSize(QtCore.QSize(1000, 25))
        self.textBrowser_4.setStyleSheet("border:1px solid rgb(218, 218, 218);\n"
"background-color: rgb(218, 218, 218);")
        self.textBrowser_4.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.horizontalLayout_2.addWidget(self.textBrowser_4)
        self.progressBar_2 = QtWidgets.QProgressBar(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar_2.sizePolicy().hasHeightForWidth())
        self.progressBar_2.setSizePolicy(sizePolicy)
        self.progressBar_2.setMinimumSize(QtCore.QSize(0, 25))
        self.progressBar_2.setMaximumSize(QtCore.QSize(1000, 25))
        self.progressBar_2.setStyleSheet("background-color: rgb(218, 218, 218);")
        self.progressBar_2.setProperty("value", 24)
        self.progressBar_2.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar_2.setObjectName("progressBar_2")
        self.horizontalLayout_2.addWidget(self.progressBar_2)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 1)
        self.horizontalLayout_2.setStretch(3, 1)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addWidget(self.frame_2)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.AvailData = QtWidgets.QAction(MainWindow)
        self.AvailData.setObjectName("AvailData")
        self.trainModel = QtWidgets.QAction(MainWindow)
        self.trainModel.setObjectName("trainModel")
        self.trainModel_2 = QtWidgets.QAction(MainWindow)
        self.trainModel_2.setObjectName("trainModel_2")
        self.www = QtWidgets.QAction(MainWindow)
        self.www.setObjectName("www")
        self.AvailModel = QtWidgets.QAction(MainWindow)
        self.AvailModel.setObjectName("AvailModel")
        self.train_Model = QtWidgets.QAction(MainWindow)
        self.train_Model.setObjectName("train_Model")
        self.openDir = QtWidgets.QAction(MainWindow)
        self.openDir.setObjectName("openDir")
        self.open_File = QtWidgets.QAction(MainWindow)
        self.open_File.setObjectName("open_File")
        self.open_Dir = QtWidgets.QAction(MainWindow)
        self.open_Dir.setObjectName("open_Dir")
        self.save_act = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/DeepCID/icon/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save_act.setIcon(icon)
        self.save_act.setObjectName("save_act")
        self.action_link = QtWidgets.QAction(MainWindow)
        self.action_link.setObjectName("action_link")
        self.action_setup = QtWidgets.QAction(MainWindow)
        self.action_setup.setObjectName("action_setup")
        self.import_act = QtWidgets.QAction(MainWindow)
        self.import_act.setObjectName("import_act")
        self.open_act = QtWidgets.QAction(MainWindow)
        self.open_act.setObjectName("open_act")

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DeepCID"))
        self.data_.setText(_translate("MainWindow", "Data"))
        self.function_.setText(_translate("MainWindow", "Function"))
        self.help_.setText(_translate("MainWindow", "Help"))
        self.database_path.setText(_translate("MainWindow", "TextLabel"))
        self.open_database.setText(_translate("MainWindow", "open"))
        self.build_database.setText(_translate("MainWindow", "build"))
        self.label.setText(_translate("MainWindow", "database"))
        self.add_group.setText(_translate("MainWindow", "add"))
        self.delete_group.setText(_translate("MainWindow", "delete"))
        self.label_2.setText(_translate("MainWindow", "group"))
        self.add_spectra.setText(_translate("MainWindow", "add"))
        self.delete_spectra.setText(_translate("MainWindow", "delete"))
        self.label_3.setText(_translate("MainWindow", "spectra"))
        self.clear_plot_area.setText(_translate("MainWindow", "clear"))
        self.collect_spectra.setText(_translate("MainWindow", "collect"))
        self.label_12.setText(_translate("MainWindow", "plot"))
        self.train_run.setText(_translate("MainWindow", "start"))
        self.label_4.setText(_translate("MainWindow", "training"))
        self.open_mix.setText(_translate("MainWindow", "open"))
        self.pred_run.setText(_translate("MainWindow", "start"))
        self.ratio_estimation.setText(_translate("MainWindow", "ratio"))
        self.save_results.setText(_translate("MainWindow", "save"))
        self.label_11.setText(_translate("MainWindow", "prediction"))
        self.show_help_html.setText(_translate("MainWindow", "html"))
        self.show_demo.setText(_translate("MainWindow", "demo"))
        self.label_5.setText(_translate("MainWindow", "help"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Components"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Unknown"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Results"))
        self.textBrowser_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.textBrowser_4.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.AvailData.setText(_translate("MainWindow", "load model"))
        self.trainModel.setText(_translate("MainWindow", "训练模型"))
        self.trainModel_2.setText(_translate("MainWindow", "train model"))
        self.www.setText(_translate("MainWindow", "construct new models"))
        self.AvailModel.setText(_translate("MainWindow", "Load models"))
        self.train_Model.setText(_translate("MainWindow", "New models"))
        self.openDir.setText(_translate("MainWindow", "dir"))
        self.open_File.setText(_translate("MainWindow", "file"))
        self.open_Dir.setText(_translate("MainWindow", "dir"))
        self.save_act.setText(_translate("MainWindow", "Save as..."))
        self.action_link.setText(_translate("MainWindow", "link to database"))
        self.action_setup.setText(_translate("MainWindow", "set up new database"))
        self.import_act.setText(_translate("MainWindow", "Import"))
        self.open_act.setText(_translate("MainWindow", "Open"))