#include "connectus.h"
#include <QLabel>

ConnectUs::ConnectUs(QWidget *parent) : QMainWindow(parent)
{
    //size
    this->setFixedSize(700,400);
    //logo
    this->setWindowIcon(QIcon(":/scene/logo.jpg"));
    QString qstr = QString::fromLocal8Bit("三星堆文物修复模型");
    //QString qstr = QString::fromLocal8Bit("三星堆文物修复模型");
    this->setWindowTitle(qstr);

    QLabel *lab= new QLabel(this);
    //lab->hide();
    lab->setGeometry(20,20,960,560);
    lab->setAlignment(Qt::AlignTop);
    lab->setText(QString::fromLocal8Bit("亲爱的用户："
                 "<br/>"
                 "   您好。如果您在使用我们软件的过程中发现了任何问题，或对本软件有"
                 "<br/>"
                 "任何建议，欢迎您联系我们。"
                 "<br/>"
                 "   E-Mail: relicsrepair@163.com"));
    lab->setStyleSheet("QLabel{font:20px}");
}
