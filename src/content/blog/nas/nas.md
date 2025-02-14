---
title: 'How to build NAS '
publishDate: 2024-10-19
description: 'build nas by a computer and HDD'
tags:
  - Technology
  - NAS
language: 'Chinese'
heroImage: { src: './nas.png', color: '#D58388' }
---
该博客主要介绍如何利用一台电脑+机械硬盘来搭建一台可以随地访问的nas

## **配置：**
mac ipad iphone作为接收，以此来随地接收文件
一台win11+ubuntu24的双系统电脑+西部数据4t紫盘机械硬盘+绿联硬盘盒

机械硬盘在刚买回来的时候要进行初始化，我用的是mac的磁盘工具对他进行“抹掉”
然后为其分配一个文件系统这里我选择exFAT，因为这个格式对不同的操作系统兼容性更强

一开时用了绿联的硬盘盒将机械硬盘装进去用usb3.0数据线连接到linux电脑上

## **Samba**
然后在linux上要安装Samba（用于文件共享）

```bash
sudo apt update
sudo apt install samba
```

用vim编辑Samba配置文件

```bash
sudo vim /etc/samba/smb.conf
```

在文件末尾添加一下内容
```shell
[MyNAS]
path = /path/to/your/harddrive//这里是你机械硬盘在你电脑的位置，可以在图形化界面打开这个机械硬盘然后再在终端中看他的路径，像我的就在/media/laurie/HDD
available = yes
valid users = laurie//这里可以自己填用户名
read only = no
browsable = yes
public = yes
writable = yes
```

然后保存退出
然后创建一个Samba用户，会要求你输入一个密码

```bash
sudo smbpasswd -a laurie//这里的用户名填上面配置的用户名
```

然后重启Samba服务

```shell
sudo systemctl restart smbd
```

## **ZeroTier**
### linux
然后进行内网穿透的话我使用ZeroTier，原理的话可以自行chatgpt
首先在linux上安装zerotier

```shell
curl -s https://install.zerotier.com | sudo bash
```

启动zerotier服务

```shell
sudo systemctl enable zerotier-one
sudo systemctl start zerotier-one
```

然后访问[zerotier](https://www.zerotier.com)创建一个账号，然后创建一个新的虚拟网络
创建完之后会得到一个NEetwork ID
这里用哪一个设备进行创建都可以
然后在linux中加入ZeroTier网络

```shell
sudo zerotier-cli join <Network ID>
```

进入 ZeroTier Central，找到刚刚加入的设备，勾选旁边的复选框，批准该设备加入网络
然后可以在那个界面看到zerotier为你分配的managed ip

### Mac
然后现在要保证你要连接的设备和你的nas在同一个网络中
在mac上用命令行安装ZeroTier（也可以下载ZeroTier app 我是用的命令行）

```shell
brew install zerotier-one
```

安装完后启动zerotier服务

```shell
sudo zerotier-cli join <Network ID>
```

在zerotier网站上检查是否已经添加成功

### ipad&iphone
在这两个设备下要下载zerotier app外区的app store有的下，然后点击+号，输入network id来加入网络

## **Finder**
最后就是利用苹果的finder来对硬盘中的文件进行访问
### MAC
command+k然后弹出连接服务器窗口
在搜索栏里填写smb://<虚拟ip>/<Samba配置文件添加内容的那个头名字，在我的上面的例子中就是MyNas>
然后填写nas的用户名和密码就可以连接

### Win
可以打开文件资源管理器，直接在地址栏中输入以下格式：

```shell
\\<你的Managed IP>\<共享文件夹>
```
没试过不保证成功率

### Ipad&iphone
在文件中找到添加服务器
然后也按照这个格式smb://<虚拟ip>/<Samba配置文件添加内容的那个头名字，在我的上面的例子中就是MyNas>填写

但是iphone好像会不是很稳定，目前还没找到解决办法


## **shell**
当然也可以用ssh远程连接
```shell
shell hostname@虚拟ip
```
就可以连接


## **总结**
至此我们就搭建了nas服务器，最后如果遇到什么问题请问chatgpt他才是世界上最好的老师