---
title: 'Note 软件开发'
publishDate: 2025-06-24
description: 'A lightweight note app for macOS'
tags:
 - 开发
language: 'Chinese'
heroImage: { src: './Note.png', color: '#5F8B92'}
---

## 前言

大概从上学期开始就有一个想法就是开发一个便签，之前找过一些市面上的便签。之前一直用的是simple antnotes，用的到还可以，只是有时用的不是很爽，之前的电脑有时莫名其妙的重启不知道原因，但是怀疑和这个软件有关系，而且我觉得这软件做的不够优雅，苹果自带的便签也好丑。而且我还想要我的便签能支持markdown，就像obsidian和typora那样支持实时编译，那样可以满足我的需求。

所以就有了一些想法，我一个学计算机的连软件都不会开发那就有点逊了。正好之前在数据库上老师讲过一些方案，主要介绍的就是js+electron，一方面他很方便，不用学习每个os的细节，基本上你只用会前端就差不多能开发。而且因为他是基于浏览器作为内核，所以基本上写一次就可以在不同的系统上跑。

本博客就记录一下一个新手小白的开发流程。
by the way，这个项目放在[这里](https://github.com/laurie-hxf/note)

## 准备

一开始就了解到js的重要性，然后选了[这个网站](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript)作为入门，感觉挺好的，挺细致的。还有就是electron的[网站](https://www.electronjs.org/docs/latest/)

js的语法感觉没有很复杂，我看了一下他的网址，看完了第一个项目就不想看了，不如直接开发，边开发边学习，主要是我觉得没有很难理解，而且这种做法效率还高。

基本上遇到什么问一下gpt，效率还是挺高的。

然后就是electron，这个我感觉挺有意思的，就简单记录一下我用到的命令

首先就是安装，这里我用了淘宝的镜像，之前一直都安装不上
```shell
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ npm install electron --save-dev 
```

快速创建一个默认的 `package.json` 文件，用于管理你的 Node 项目依赖、脚本、版本等信息。
```shell
npm init -y  
```

这个命令本质是运行`package.json`中`scripts.start`的部分，取决于你的`package.json`部分怎么写
```json
"scripts": {
  "start": "electron ."
}
```
然后等价于运行electron .这个命令，它会用 Electron 启动你当前的项目目录，作为一个桌面应用运行。可以用来实时看你的软件。
```shell
npm run start 
```

要是想要将你的软件打包成一个app之类的话，就要安装electron-builder
```shell
npm install electron-builder --save-dev    
```

要想能在mac上打包成win的exe文件，就要先安装这个虚拟环境
```shell
brew install --cask --no-quarantine wine-stable    
```

用来将软件打包成win-x64架构的指令，如果不加后面的x64，他就会默认使用当前电脑的架构，像我就是ARM架构的
```shell
npx electron-builder --win --x64   
```

运行这个命令的话就是运行 `scripts.build` 定义的命令，比如打包构建。本质上就是在运行electron-builder这个命令，然后顶层 `"build"` 部分就是具体的配置
```shell
npm run build  
```

下面贴一个我的`package.json`
```json
{
"devDependencies": {
"electron": "^36.5.0",
"electron-builder": "^26.0.12"
},
"name": "note",
"version": "1.0.0",
"main": "main.js",
"dependencies": {
"boolean": "^3.2.0",
"buffer-crc32": "^0.2.13",
"cacheable-lookup": "^5.0.4",
"cacheable-request": "^7.0.4",
"clone-response": "^1.0.3",
"debug": "^4.4.1",
"decompress-response": "^6.0.0",
"defer-to-connect": "^2.0.1",
"define-data-property": "^1.1.4",
"define-properties": "^1.2.1",
"detect-node": "^2.1.0",
"end-of-stream": "^1.4.5",
"env-paths": "^2.2.1",
"es-define-property": "^1.0.1",
"es-errors": "^1.3.0",
"es6-error": "^4.1.1",
"escape-string-regexp": "^4.0.0",
"extract-zip": "^2.0.1",
"fd-slicer": "^1.1.0",
"fs-extra": "^8.1.0",
"get-stream": "^5.2.0",
"global-agent": "^3.0.0",
"globalthis": "^1.0.4",
"gopd": "^1.2.0",
"got": "^11.8.6",
"graceful-fs": "^4.2.11",
"has-property-descriptors": "^1.0.2",
"http-cache-semantics": "^4.2.0",
"http2-wrapper": "^1.0.3",
"json-buffer": "^3.0.1",
"json-stringify-safe": "^5.0.1",
"jsonfile": "^4.0.0",
"keyv": "^4.5.4",
"lowercase-keys": "^2.0.0",
"matcher": "^3.0.0",
"mimic-response": "^1.0.1",
"ms": "^2.1.3",
"normalize-url": "^6.1.0",
"object-keys": "^1.1.1",
"once": "^1.4.0",
"p-cancelable": "^2.1.1",
"pend": "^1.2.0",
"progress": "^2.0.3",
"pump": "^3.0.3",
"quick-lru": "^5.1.1",
"resolve-alpn": "^1.2.1",
"responselike": "^2.0.1",
"roarr": "^2.15.4",
"semver": "^6.3.1",
"semver-compare": "^1.0.0",
"serialize-error": "^7.0.1",
"sprintf-js": "^1.1.3",
"sumchecker": "^3.0.1",
"type-fest": "^0.13.1",
"undici-types": "^6.21.0",
"universalify": "^0.1.2",
"wrappy": "^1.0.2",
"yauzl": "^2.10.0"
},
"scripts": {
"test": "echo \"Error: no test specified\" && exit 1",
"start": "electron .",
"build": "electron-builder"
},
"build": {
"appId": "com.yourname.guessgame",
"productName": "Note",
"icon": "./icon",
"files": [
"**/*"
],
"directories": {
"output": "dist"
},
"mac": {
"target": "dmg"
},
"win": {
"target": "nsis"
},
"linux": {
"target": "AppImage"
}
},
"keywords": [],
"author": "",
"license": "ISC",
"description": ""
}
```


## 进展

目前大概就是一个普通的便签，还没有支持markdown，也没有实时编译，然后这个项目的大部份代码还是ai写的，前端css那些，~~一点没有设计天赋~~，包括一些js，所以还要去学习这些代码后续再进一步开发。而且window和mac上都有一些小bug，但不影响使用，后续还要进一步调整。

## 发布

初步完成之后，就上架了github，发布了release，加了一下开源协议，又学到新技能。

然后就是将一些视频可以加进README里面，准确来说不是视频，而是gif，markdown支持播放gif。然后下面的命令就是将视频转成gif。ffmpeg nb，[bellard](https://bellard.org/) nb。
```shell
ffmpeg -i 录屏2025-06-23\ 20.01.35.mov -vf "fps=15" demo2.gif       
```

还有就是一个工具可以把你按的键位显示在屏幕上，[这个工具](https://github.com/keycastr/keycastr)也挺出名的
```shell
brew install --cask keycastr    
```