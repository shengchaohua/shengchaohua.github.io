---
title: 数学
order: 6
---

<!-- more -->

## Leetcode 编程题

### LCR 187. 圆圈中最后剩下的数字

> [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/ "剑指 Offer 62. 圆圈中最后剩下的数字")

一、题目

0,1,...,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

二、解析

> 参考 [LeetCode-Solution](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/solution/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-by-lee/ "LeetCode-Solution")
> 
> 参考 [约瑟夫环——公式法（递推公式）](https://blog.csdn.net/u011500062/article/details/72855826 "约瑟夫环——公式法（递推公式）")

1）使用数组模拟，超时。

2）数学+递归。代码如下：

```python
# Python 默认的递归深度不够，需要手动设置
sys.setrecursionlimit(100000)

class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        def f(n, m):
            if n == 0:
                return 0
            x = f(n - 1, m)
            return (m + x) % n
        
        return f(n, m)
```

3）递推公式

```python
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        res = 0
        for i in range(2, n + 1):
            res = (res + m) % i
        return res
```



### 172. 阶乘后的零

> [172. 阶乘后的零](https://leetcode-cn.com/problems/factorial-trailing-zeroes/ "172. 阶乘后的零")

一、题目

给定一个整数 `n` ，返回 `n!` 结果中尾随零的数量。

提示：`n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1`

**示例 1：**

```text
输入：n = 3
输出：0
解释：3! = 6 ，不含尾随 0
```

二、解析

代码如下：

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        res = 0
        while n >= 5:
            res += n // 5
            n //= 5
        return res
```



### 233. 数字 1 的个数

> [233. 数字 1 的个数](https://leetcode-cn.com/problems/number-of-digit-one/ "233. 数字 1 的个数")

一、题目

给定一个整数 n，计算所有小于等于 n 的非负整数中数字 1 出现的个数。

示例:

```纯文本
输入: 13
输出: 6 
解释: 数字 1 出现在以下数字中: 1, 10, 11, 12, 13 。
```

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/number-of-digit-one/solution/shu-zi-1-de-ge-shu-by-leetcode/ "Leetcode官方题解")

代码如下：

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        if n <= 0:
            return 0
        count = 0
        i = 1
        while n // i != 0:
            high = n // (10 * i)
            cur = (n // i) % 10
            low = n - (n // i) * i
            if cur == 0:
                count += high * i
            elif cur == 1:
                count += high * i + ( low + 1)
            else:
                count += (high + 1) * i
            i = i * 10
        return count
```



### 面试题 08.04. 幂集

> [面试题 08.04. 幂集](https://leetcode-cn.com/problems/power-set-lcci/ "面试题 08.04. 幂集")

一、题目

幂集。编写一种方法，返回某集合的所有子集。集合中不包含重复的元素。

说明：解集不能包含重复的子集。

二、解析

1）使用位图，每个二进制位是否选择一个数字，0表示不选择，1表示选择。

代码如下：

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        output = []
        
        for i in range(2**n, 2**(n + 1)):
            # generate bitmask, from 0..00 to 1..11
            bitmask = bin(i)[3:]
            output.append([nums[j] for j in range(n) if bitmask[j] == '1'])
        
        return output
```

2）DFS

与78相同。



### 343. 整数拆分

> [343. 整数拆分](https://leetcode-cn.com/problems/integer-break/ "343. 整数拆分")

一、题目

给定一个正整数 n，将其拆分为**至少**两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

二、解析

> 参考 [Leetcode-Krahets](https://leetcode-cn.com/problems/integer-break/solution/343-zheng-shu-chai-fen-tan-xin-by-jyd/ "Leetcode-Krahets")

把整数拆分出较多的3，注意4需要拆分成两个2。

代码如下：

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        if n <= 3:
            return n - 1
        if n % 3 == 0:
            return 3 ** (n//3)
        if n % 3 == 1:
            return 3 ** (n//3 - 1) * 2 * 2
        return 3 ** (n//3) * 2
```



### 470. 用 Rand7() 实现 Rand10()

> [470. 用 Rand7() 实现 Rand10()](https://leetcode-cn.com/problems/implement-rand10-using-rand7/ "470. 用 Rand7() 实现 Rand10()")

一、题目

已有方法 rand7 可生成 1 到 7 范围内的均匀随机整数，试写一个方法 rand10 生成 1 到 10 范围内的均匀随机整数。

不要使用系统的 Math.random() 方法。

二、解析

代码如下：

```python
class Solution:
    def rand10(self):
        res = 49
        while res > 40:
            res = (rand7() - 1) * 7 + rand7() # 1 - 40
        return (res - 1) % 10 + 1
```



### 470-1. 用 Rand3() 实现 Rand5()

代码如下：

```python
class Solution:
    def rand5(self):
        res = 9
        while res > 5:
            res = (rand3() - 1) * 3 + rand3() # 1 - 5
        return res
```



### 470-2. 用 Rand5() 实现 Rand7()

代码如下：

```python
class Solution:
    def rand7(self):
        res = 9
        while res > 21:
            res = (rand5() - 1) * 5 + rand5() # 1 - 21
        return (res - 1) % 7 + 1
```



### 223. 矩形面积

> [223. 矩形面积](https://leetcode-cn.com/problems/rectangle-area/ "223. 矩形面积")

一、题目

在二维平面上计算出两个由直线构成的矩形重叠后形成的总面积。

每个矩形由其左下顶点和右上顶点坐标表示，如图所示。

二、解

代码如下：

```python
class Solution:
    def computeArea(self, A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int) -> int:
        area1 = (C - A) * (D - B)
        area2 = (G - E) * (H - F)
        if E >= C or F >= D or A >= G or B >= H:
            return area1 + area2
        cross_area = (min(C, G) - max(A, E)) * (min(D, H) - max(B, F))
        return area1 + area2 - cross_area
```



### 264. 丑数 II

> [264. 丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/ "264. 丑数 II")

一、题目

编写一个程序，找出第 n 个丑数。

丑数就是质因数只包含 2, 3, 5 的正整数。

示例:

```纯文本
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
```

二、解析

代码如下：

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        num_list = [1]
        i2 = i3 = i5 = 0
        while len(num_list) < n:
            num2 , num3, num5 = num_list[i2] * 2, num_list[i3] * 3, num_list[i5] * 5
            num = min(num2, num3, num5)
            if num == num2:
                i2 += 1
            if num == num3:
                i3 += 1
            if num == num5:
                i5 += 1
            num_list.append(num)
        return num_list[-1]
```



### 191. 二进制中1的个数

> [191. 位1的个数](https://leetcode-cn.com/problems/number-of-1-bits/ "191. 位1的个数")

一、题目

编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。

提示：

-   请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
-   在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在 示例 3 中，输入表示有符号整数 -3。

示例 1：

```text
输入：n = 00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
```

示例 2：

```text
输入：n = 00000000000000000000000010000000
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
```

示例 3：

```text
输入：n = 11111111111111111111111111111101
输出：31
解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。
```

二、解析

使用位运算。代码如下：

```go
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n > 0:
            if n & 1 == 1:
                count += 1
            n >>= 1
        return count
```



### 670. 最大交换

> [670. 最大交换](https://leetcode.cn/problems/maximum-swap/)

一、题目

给定一个非负整数，你**至多**可以交换一次数字中的任意两位。返回你能得到的最大值。

**示例 1 :**

```text
输入: 2736
输出: 7236
解释: 交换数字2和数字7。
```

**示例 2 :**

```纯文本
输入: 9973
输出: 9973
解释: 不需要交换。
```

**注意：** 给定数字的范围是 \[0, 108]

二、解析

> [官方题解](https://leetcode.cn/problems/maximum-swap/solutions/1818457/zui-da-jiao-huan-by-leetcode-solution-lnd5/)

1）直接遍历。

由于对于整数 num 的十进制数字位长最长为 8 位，任意两个数字交换一次最多有 28 种不同的交换方法，因此我们可以尝试遍历所有可能的数字交换方法即可，并找到交换后的最大数字即可。

代码如下：

```python
class Solution:
    def maximumSwap(self, num: int) -> int:
        ans = num
        s = list(str(num))
        for i in range(len(s)):
            for j in range(i):
                s[i], s[j] = s[j], s[i]
                ans = max(ans, int(''.join(s)))
                s[i], s[j] = s[j], s[i]
        return ans
```

2）贪心

可以观察到右边越大的数字与左边较小的数字进行交换，这样产生的整数才能保证越大。因此我们可以利用贪心法则，尝试将数字中右边较大的数字与左边较小的数字进行交换，这样即可保证得到的整数值最大。：

具体做法如下：

-   我们将从右向左扫描数字数组，并记录当前已经扫描过的数字的最大值的索引maxId，且保证 maxId 越靠近数字的右侧。
-   如果检测到当前数字 charArray\[i]\<charArray\[maxId]，此时则说明索引 i 的右侧的数字最大值为charArray\[maxId]，此时我们可以尝试将charArray\[i] 与charArray\[maxId] 进行交换即可得到一个比 num更大的值。

代码如下：

```python
class Solution:
    def maximumSwap(self, num: int) -> int:
        s = list(str(num))
        n = len(s)
        maxIdx = n - 1
        idx1 = idx2 = -1
        for i in range(n - 1, -1, -1):
            if s[i] > s[maxIdx]:
                maxIdx = i
            elif s[i] < s[maxIdx]:
                idx1, idx2 = i, maxIdx
        if idx1 < 0:
            return num
        s[idx1], s[idx2] = s[idx2], s[idx1]
        return int(''.join(s))

```



### 1502. 判断能否形成等差数列

> [1502. 判断能否形成等差数列](https://leetcode.cn/problems/can-make-arithmetic-progression-from-sequence/)

一、题目

给你一个数字数组 arr 。

如果一个数列中，任意相邻两项的差总等于同一个常数，那么这个数列就称为 等差数列 。

如果可以重新排列数组形成等差数列，请返回 true ；否则，返回 false 。

示例 1：

```text
输入：arr = [3,5,1]
输出：true
解释：对数组重新排序得到 [1,3,5] 或者 [5,3,1] ，任意相邻两项的差分别为 2 或 -2 ，可以形成等差数列。
```

示例 2：

```text
输入：arr = [1,2,4]
输出：false
解释：无法通过重新排序得到等差数列。
```

二、解析

1）排序。

非常直观，代码如下：

```python
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        if len(arr) <= 2:
            return True
        
        arr.sort()
        target = arr[1] - arr[0]
        for i in range(2, len(arr)):
            if arr[i] - arr[i - 1] != target:
                return False
        return True

```

分析：时间复杂度为O(nlogn)。

2）数学方法，寻找等差数列的规律。如果是等差数列，假设最小值为a，公差为n，那么数列中为a+n，...，a+x\*n。

> [官方题解-评论](https://leetcode.cn/problems/can-make-arithmetic-progression-from-sequence/solution/pan-duan-neng-fou-xing-cheng-deng-chai-shu-lie-by-/1252762)

评论：1. 先遍历第一遍，得出最大值和最小值。 2. 如果最大值最小值相等，则肯定是等差数列。 3. 否则（最大值大于最小值）最大值和最小值的差记为diff。 3.1 如果diff不能被nums.length - 1整除，不是等差数列。 3.2 如果能整除，商x就是公差。 3.2.1 再遍历第二遍，看每个数减去最小值能不能被x整除，不能的话，不是等差数列。 3.2.2 能整除的话，商可以表示这个元素是等差数列的第几个元素。用一个临时数组（长度也是nums.length）标记一下这第几个元素是否出现过，如果出现过就有重复值，不是等差数列。 3.2.3 如果每个元素都是唯一的，则是等差数列。	

代码如下：

```go
func canMakeArithmeticProgression(arr []int) bool {
	if len(arr) <= 2 {
		return true
	}

	minNum := arr[0]
	maxNum := arr[0]
	for _, num := range arr {
		if num < minNum {
			minNum = num
		}
		if num > maxNum {
			maxNum = num
		}
	}

	maxMinDiff := maxNum - minNum
	if maxMinDiff == 0 {
		return true
	}

	count := len(arr)
	if maxMinDiff%(count-1) != 0 {
		return false
	}

	step := maxMinDiff / (count - 1)
	flags := make([]int, count)
	for _, num := range arr {
		temp := num - minNum
		if temp%step != 0 {
			return false
		}
		numIdx := temp / step
		if flags[numIdx] == 1 {
			return false
		}
		flags[numIdx] = 1
	}

	return true
}
```



### 166. 分数到小数

> [166. 分数到小数](https://leetcode.cn/problems/fraction-to-recurring-decimal/)

一、题目

给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以 字符串形式返回小数 。

如果小数部分为循环小数，则将循环的部分括在括号内。

如果存在多个答案，只需返回 任意一个 。

对于所有给定的输入，保证答案字符串的长度小于 10^4 。

示例 1：

```text
输入：numerator = 1, denominator = 2
输出："0.5"
```

示例 2：

```text
输入：numerator = 2, denominator = 1
输出："2"
```

示例 3：

```text
输入：numerator = 4, denominator = 333
输出："0.(012)"
```

二、解析

> [官方题解](https://leetcode.cn/problems/fraction-to-recurring-decimal/solutions/1028368/fen-shu-dao-xiao-shu-by-leetcode-solutio-tqdw/)

如果分子可以被分母整除，则结果是整数，将分子除以分母的商以字符串的形式返回即可。

如果分子不能被分母整除，则结果是有限小数或无限循环小数，需要通过模拟长除法的方式计算结果。为了方便处理，首先根据分子和分母的正负决定结果的正负（注意此时分子和分母都不为 0），然后将分子和分母都转成正数，再计算长除法。

计算长除法时，首先计算结果的整数部分，将以下部分依次拼接到结果中：

-   如果结果是负数则将负号拼接到结果中，如果结果是正数则跳过这一步；
-   将整数部分拼接到结果中；
-   将小数点拼接到结果中。

完成上述拼接之后，根据余数计算小数部分。

计算小数部分时，每次将余数乘以 10，然后计算小数的下一位数字，并得到新的余数。重复上述操作直到余数变成 0或者找到循环节。

-   如果余数变成 0，则结果是有限小数，将小数部分拼接到结果中。
-   如果找到循环节，则找到循环节的开始位置和结束位置并加上括号，然后将小数部分拼接到结果中。

如何判断是否找到循环节？注意到对于相同的余数，计算得到的小数的下一位数字一定是相同的，因此如果计算过程中发现某一位的余数在之前已经出现过，则为找到循环节。为了记录每个余数是否已经出现过，需要使用哈希表存储每个余数在小数部分第一次出现的下标。

Python代码如下：

```python
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        if numerator % denominator == 0:
            return str(numerator // denominator)

        s = []
        if (numerator < 0) != (denominator < 0):
            s.append('-')

        # 整数部分
        numerator = abs(numerator)
        denominator = abs(denominator)
        integerPart = numerator // denominator
        s.append(str(integerPart))
        s.append('.')

        # 小数部分
        indexMap = {}
        remainder = numerator % denominator
        while remainder and remainder not in indexMap:
            indexMap[remainder] = len(s)
            remainder *= 10
            s.append(str(remainder // denominator))
            remainder %= denominator
        if remainder > 0:  # 有循环节
            insertIndex = indexMap[remainder]
            s.insert(insertIndex, '(')
            s.append(')')

        return ''.join(s)
```

Golang代码如下：

```go
func fractionToDecimal(numerator, denominator int) string {
    if numerator%denominator == 0 {
        return strconv.Itoa(numerator / denominator)
    }

    s := []byte{}
    if numerator < 0 != (denominator < 0) {
        s = append(s, '-')
    }

    // 整数部分
    numerator = abs(numerator)
    denominator = abs(denominator)
    integerPart := numerator / denominator
    s = append(s, strconv.Itoa(integerPart)...)
    s = append(s, '.')

    // 小数部分
    indexMap := map[int]int{}
    remainder := numerator % denominator
    for remainder != 0 && indexMap[remainder] == 0 {
        indexMap[remainder] = len(s)
        remainder *= 10
        s = append(s, '0'+byte(remainder/denominator))
        remainder %= denominator
    }
    if remainder > 0 { // 有循环节
        insertIndex := indexMap[remainder]
        s = append(s[:insertIndex], append([]byte{'('}, s[insertIndex:]...)...)
        s = append(s, ')')
    }

    return string(s)
}

func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}
```



### 150. 逆波兰表达式求值

> [150. 逆波兰表达式求值](https://leetcode.cn/problems/evaluate-reverse-polish-notation/)

一、题目

> [https://leetcode.cn/problems/evaluate-reverse-polish-notation/](https://leetcode.cn/problems/evaluate-reverse-polish-notation/ "https://leetcode.cn/problems/evaluate-reverse-polish-notation/")

给你一个字符串数组 tokens ，表示一个根据 逆波兰表示法 表示的算术表达式。

请你计算该表达式。返回一个表示表达式值的整数。

注意：

有效的算符为 '+'、'-'、' \*' 和 '/' 。
每个操作数（运算对象）都可以是一个整数或者另一个表达式。
两个整数之间的除法总是 向零截断 。
表达式中不含除零运算。
输入是一个根据逆波兰表示法表示的算术表达式。
答案及所有中间计算结果可以用 32 位 整数表示。

示例 1：

```text
输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
```

示例 2：

```纯文本
输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
```

示例 3：

```text
输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
解释：该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

二、解析

使用栈。

代码如下：

```go
func evalRPN(tokens []string) int {
    stack := []int{}
    for _, token := range tokens {
        val, err := strconv.Atoi(token)
        if err == nil {
            stack = append(stack, val)
        } else {
            num1, num2 := stack[len(stack)-2], stack[len(stack)-1]
            stack = stack[:len(stack)-2]
            switch token {
            case "+":
                stack = append(stack, num1+num2)
            case "-":
                stack = append(stack, num1-num2)
            case "*":
                stack = append(stack, num1*num2)
            default:
                stack = append(stack, num1/num2)
            }
        }
    }
    return stack[0]
}

```

### 224. 基本计算器

> [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/ "224. 基本计算器")

一、题目

实现一个基本的计算器来计算一个简单的字符串表达式的值。
字符串表达式可以包含左括号`(`，右括号`)`，加号`+`，减号`-`，非负整数和空格` `。

二、解析

==TODO==



### 227. 基本计算器II

> [227. 基本计算器 II](https://leetcode.cn/problems/basic-calculator-ii/)



### 282. 为运算表达式设计优先级

> [282. 给表达式添加运算符](https://leetcode.cn/problems/expression-add-operators/)

一、题目

给你一个由数字和运算符组成的字符串 expression ，按不同优先级组合数字和运算符，计算并返回所有可能组合的结果。你可以 按任意顺序 返回答案。

生成的测试用例满足其对应输出值符合 32 位整数范围，不同结果的数量不超过 104 。

示例 1：

```text
输入：expression = "2-1-1"
输出：[0,2]
解释：
((2-1)-1) = 0 
(2-(1-1)) = 2
```

示例 2：

```text
输入：expression = "23-45"
输出：[-34,-14,-10,-10,10]
解释：
(2*(3-(45))) = -34
((23)-(45)) = -14
((2(3-4))5) = -10
(2((3-4)5)) = -10
(((23)-4)*5) = 10
```

提示：

-   1 <= expression.length <= 20
-   expression 由数字和算符 '+'、'-' 和 ' \*' 组成。
-   输入表达式中的所有整数值在范围 \[0, 99]&#x20;

二、解析

递归搜索。

代码如下：

```go
const addition, subtraction, multiplication = -1, -2, -3

func diffWaysToCompute(expression string) []int {
	ops := []int{}
	for i, n := 0, len(expression); i < n; {
		if unicode.IsDigit(rune(expression[i])) {
			x := 0
			for ; i < n && unicode.IsDigit(rune(expression[i])); i++ {
				x = x*10 + int(expression[i]-'0')
			}
			ops = append(ops, x)
		} else {
			if expression[i] == '+' {
				ops = append(ops, addition)
			} else if expression[i] == '-' {
				ops = append(ops, subtraction)
			} else {
				ops = append(ops, multiplication)
			}
			i++
		}
	}

	n := len(ops)
	dp := make([][][]int, n)
	for i := range dp {
		dp[i] = make([][]int, n)
	}

	return dfs(ops, 0, n-1, dp)
}

func dfs(ops []int, l, r int, dp [][][]int) []int {
	res := dp[l][r]
	if res != nil {
		return res
	}
	if l == r {
		dp[l][r] = []int{ops[l]}
		return dp[l][r]
	}
	for i := l; i < r; i += 2 {
		left := dfs(ops, l, i, dp)
		right := dfs(ops, i+2, r, dp)
		for _, x := range left {
			for _, y := range right {
				if ops[i+1] == addition {
					dp[l][r] = append(dp[l][r], x+y)
				} else if ops[i+1] == subtraction {
					dp[l][r] = append(dp[l][r], x-y)
				} else {
					dp[l][r] = append(dp[l][r], x*y)
				}
			}
		}
	}
	return dp[l][r]
}
```



### 282. 给表达式添加运算符

> [282. 给表达式添加运算符](https://leetcode.cn/problems/expression-add-operators/)

一、题目

给定一个仅包含数字 0-9 的字符串 num 和一个目标值整数 target ，在 num 的数字之间添加 二元 运算符（不是一元）+、- 或 \* ，返回 所有 能够得到 target 的表达式。

注意，返回表达式中的操作数 不应该 包含前导零。

示例 1:

```text
输入: num = "123", target = 6
输出: ["1+2+3", "123"]
解释: “123” 和 “1+2+3” 的值都是6。
```

示例 2:

```text
输入: num = "232", target = 8
输出: ["23+2", "2+32"]
解释: “23+2” 和 “2+32” 的值都是8。
```

示例 3:

```text
输入: num = "3456237490", target = 9191
输出: []
解释: 表达式 “3456237490” 无法得到 9191 
```

二、解析

TODO
