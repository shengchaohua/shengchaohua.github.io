---
title: 其他
---

<!-- more -->

## Leetcode 编程题

### 146. LRU 缓存

> [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)

一、题目

请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。

实现 LRUCache 类：

-   LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
-   int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
-   void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。

函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

示例：

```text
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
```

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/lru-cache/solution/lruhuan-cun-ji-zhi-by-leetcode-solution/ "Leetcode官方题解")

可以使用有序字典。但是，在面试中，面试官一般会期望面试者能够自己实现一个简单的双向链表，而不是使用语言自带的、封装好的数据结构，哈希表可以使用语言自带的。

1）使用有序字典。

```python
class LRUCache:
    def __init__(self, capacity: int):
        from collections import OrderedDict
        self.capacity = capacity
        self.cache = OrderedDict()
        
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

2）使用自带的哈希表和自己实现的双向链表。

```python
class DLinkedNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.cache = dict()
        # 使用伪头部和伪尾部节点    
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # 如果 key 存在，先通过哈希表定位，再移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value
        
    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            # 如果 key 不存在，创建一个新的节点
            node = DLinkedNode(key, value)
            # 添加进哈希表
            self.cache[key] = node
            # 添加至双向链表的头部
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                # 如果超出容量，删除双向链表的尾部节点
                removed = self.removeTail()
                # 删除哈希表中对应的项
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
    
    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        
    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)
        
    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node
```



### 295. 数据流的中位数

> [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)

一、题目

中位数是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

-   例如 arr = \[2,3,4] 的中位数是 3 。
-   例如 arr = \[2,3] 的中位数是 (2 + 3) / 2 = 2.5 。

实现 MedianFinder 类:

-   MedianFinder() 初始化 MedianFinder 对象。
-   void addNum(int num) 将数据流中的整数 num 添加到数据结构中。
-   double findMedian() 返回到目前为止所有元素的中位数。与实际答案相差 10-5 以内的答案将被接受。

**示例 1：**

```
输入
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
输出
[null, null, null, 1.5, null, 2.0]

解释
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // 返回 1.5 ((1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
```

二、解析

1）保持数组有序，添加数据时使用二分搜索。

```python
class MedianFinder:
    def __init__(self):
        self.nums = []
        
    def addNum(self, num: int) -> None:
        import bisect
        bisect.insort(self.nums, num)
        
    def findMedian(self) -> float:
        l = len(self.nums)
        num1 = self.nums[(l - 1) // 2]
        num2 = self.nums[l // 2]
        return (num1 + num2) / 2
```

复杂度分析：

-   时间复杂度：二分搜索花费`O(log n)`，插入元素可能需要花费`O(n)`，总共是`O(n)`。
-   空间复杂度：`O(n)`。

2）使用两个堆，一个最大堆，一个最小堆。最大堆用于存储输入数字中较小的一半，方便获得其中的最大值，最小堆用于存储输入数字的较大的一半，方便获得其中的最小值。

代码如下：

```python
from heapq import *

class MedianFinder:
    def __init__(self):
        self.min_heap = []  # keep the larger half numbers
        self.max_heap = []  # keep the smaller half numbers
        
    def addNum(self, num: int) -> None:
        heappush(self.min_heap, num)
        target = heappop(self.min_heap)
        heappush(self.max_heap, -target)
        if len(self.min_heap) < len(self.max_heap):
            target = heappop(self.max_heap)
            heappush(self.min_heap, -target)
            
    def findMedian(self) -> float:
        if len(self.min_heap) == len(self.max_heap):
            return (self.min_heap[0] - self.max_heap[0]) / 2
        return self.min_heap[0]
```

复杂度分析：

-   时间复杂度：最坏情况下，从顶部有三个堆插入和两个堆删除。每一个都需要花费`O(log n)`，所以最终是`O(log n)`。
-   空间复杂度：`O(n)`。



