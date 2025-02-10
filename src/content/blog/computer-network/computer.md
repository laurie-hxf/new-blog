---
title: '离散数学'
publishDate: 2025-2-10
description: '3D imagery has the power to bring cinematic visions to life and help accurately plan tomorrow’s cityscapes. Here, 3D expert Ricardo Ortiz explains how it works.'
tags:
  - Example
  - 3D
language: 'English'
heroImage: { src: './thumbnail.jpg', color: '#D58388' }
---

## Cardinality of sets(基数)

A和B有相同的基数如果A和B之间是双射的（bijection）->表示为|A|=|B|
## Schröder-Bernstein Theorem
如果有一个单射函数 f:$A \rightarrow B$ 和一个单射函数 g：$B \rightarrow A$
那么就可以说明存在一个双射函数在A和B之间 （A和B都可以是无穷的）->|A|=|B|
$$
例子 
$$
$$f: (0,1) \rightarrow (0,1]\quad f(x)=x$$
$$g:(0,1] \rightarrow (0,1)\quad g(x)=\frac{x}{2}$$

## countable and uncountable sets

countable:有限集合或者集合的基数和$Z^+$一样
uncountable：反之就是uncountable

## 如何证明一个集合是countable：
1. 证明存在两个单射函数从$Z^+ \rightarrow A$以及从$A \rightarrow Z^+$
2. 直接列出一个序列
   <mark>eg1</mark> :证明Z是countable
   $list a sequence：0,1,-1,2,-2...$
   <mark>eg2</mark> :证明有理数是countable
>   列出所有可能
    $\frac{1}{1}\quad \frac{2}{1}\quad \frac{3}{1}\quad \frac{4}{1}\quad\frac{5}{1}\quad ...$
	$\frac{1}{2}\quad \frac{2}{2}\quad \frac{3}{2}\quad \frac{4}{3}\quad\frac{4}{1}\quad ...$
    $\frac{1}{3}\quad \frac{2}{3}\quad \frac{3}{3}\quad \frac{4}{3}\quad\frac{5}{3}\quad ...$

#### 如何证明一个集合是uncountable
$Cantor’s\: diagonal\: argument$

>Assume that R is countable.
 Then, every subset of R is countable . In particular, interval[0, 1] is countable. This implies that there exists a list $r_1, r_2, r_3, …$that can enumerate all elements in this set, where 
 $r_1 =\textcolor{red}{0.d_{11}} d_{12} d_{13}...$
 $r_2 =0.d_{21} \textcolor{red}{d_{22}} d_{23}...$
 $r_3 =0.d_{31} d_{32} \textcolor{red}{d_{33}}...$
$...$
>然后构造一个使得他的他的第i个数和$r_i$的第i个元素都不一样

<mark>eg</mark>证明𝒫(N)的power set是uncountable
Proof by contradiction: (Cantor’s diagonal argument)
• Assume that 𝒫(N) is countable.
This means that all elements of this set can be listed as $S_0, S_1, S_2, …$,where $S_i∈ \mathcal{P}(N)$. Then, each $S_i⊆ N$ can be represented by a bit string $b_{i0}b_{i1}b_{i2}···$, 
where $b_{ij}= 1$  if  $j ∈ S_i \quad and\quad b_{ij}=0 \quad if \quad j ∉ Si:$
$S_0= \textcolor{red} {b_{00}}b_{01}b_{02}b_{03}...$
$S_1=  b_{00}\textcolor{red}{b_{01}}b_{02}b_{03}...$
$S_2=  b_{20}b_{21}\textcolor{red}{b_{22}}b_{23}...$
$...$
大致意思就是将N的子集$S_i$用0101序列表示，1表示该子集存在某个元素
然后N的所有子集就可以变为一个01序列，然后用$Cantor’s\: diagonal\: argument$证明存在一个$S_i$不在那个序列中