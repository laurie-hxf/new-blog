---
title: 'The details about this blog'
publishDate: 2025-02-15
description: 'The details about this blog'
tags:
 - blog
language: 'Chinese'
heroImage: { src: './blog.png', color: '#64574D' }
---
## 动机

一开始我的博客是github pages中的[al-folio](https://github.com/alshedivat/al-folio)主题，用了一个学期之后发现之前的这个配置维护很麻烦，每次部署文章的时候都要等半天才可以部署好。所以就有了迁移博客的念头，后来就发现现在这个更好看的这个[主题](https://github.com/cworld1/astro-theme-pure)。探索了一下，这主题功能又多又好玩，耐看，部署简单，还可以实时编辑，所以就选这个主题。

## 部署
### 克隆

一开始的话克隆这个仓库，在命令行输入

```shell
git clone https://github.com/cworld1/astro-theme-pure.git
cd astro-theme-pure
```

然后我这里选择的是vercel来部署我的网站，这时建议fork这个原仓库，然后再克隆到本地，在vercel中就直接导入对应github仓库，接着就按照vercel的指示就可以。数据库我暂时还在用[leancloud](https://console.leancloud.app/apps).

然后mac用户就要下载一下bun

```shell
curl -fsSL https://bun.sh/install | bash
```

当然homebrew也可以，不过有点慢

```shell
brew install bun
```



### 运行

下载完成之后就开始安装必要软件

```shell
bun install
```

安装完之后就开始运行

```shell
bun dev
```

此时他就应该弹出local network
然后在浏览器打开连接就可以实时看自己的博客

## 配置
### 基本
在/src/site.config.ts中可以配置基本的信息，包括头像，网站名字，favicon这些在注释里都有提到，包括备案信息，github账号这些。然后在这里管理子页面，可以省去一些不要的子页面。

### 主页
这个在/src/pages/index.astro文件里面更改，可以添加一些自己的功能。

### About
这个在/src/pages/about/index.astro文件里面更改，最好玩的就是这个tool，不过这里的图片都是svg的图片，自己对应的工具要在网上找然后转化为svg图片。有一些免费的网站，比如这个[网站](https://www.vectorizer.io)就挺好用的，有一定的免费额度，但是不多。或者直接找某些软件的svg图片。

![alt text](./截屏2025-02-15%2000.03.29.png)

### Projects
这里就在/src/pages/projects/index.astro里面更改，添加自己的仓库。这个界面会展示你的GitHub贡献图，有的时候连接不上github他就会一直报错。这时候只用把那段代码暂时先注释掉该别的先。

### links
这个就在/src/pages/index/index.astro里面更改

### Blog
写博客在/src/content/blog里面添加文件夹，然后把图片放到文件夹里面。仿照现有的blog模仿就行。如果习惯在obsidian中写博客，可能格式不能兼容，可以看看我写的[工具](https://github.com/laurie-hxf/obsidian-convert-format)来转化格式。


**值得注意的是，除了blog里的图片放到对应文件夹里面，其他额外的图片都要放到/public这个文件夹里面。**

