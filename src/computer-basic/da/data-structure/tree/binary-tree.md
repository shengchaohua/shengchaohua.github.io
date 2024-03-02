---
title: 二叉树
order: 2
---

<!-- more -->

## 基础

### 树节点

1）Python

```python
class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```



## Leetcode 编程题

###  144. 二叉树的前序遍历

> [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/ "144. 二叉树的前序遍历")

遍历顺序：父节点，左子树，右子树。

1）递归版本

```python
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if root:
            res.append(root.val)
            left = self.preorderTraversal(root.left)
            res.extend(left)
            right = self.preorderTraversal(root.right)
            res.extend(right)
        return res
```

2）非递归版本

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        res = []
        stack = []
        node = root
        
        while stack or node:
            if node:
                res.append(node.val)
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                node = node.right
        return res
```

3）非递归版本

```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        res = []
        stack = [root]
        
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
```



###  94. 二叉树的中序遍历

> [94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/ "94. 二叉树的中序遍历")

遍历顺序：左子树，父节点，右子树。

1）递归版本

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if root:
            left = self.inorderTraversal(root.left)
            res.extend(left)
            res.append(root.val)
            right = self.inorderTraversal(root.right)
            res.extend(right)
        return res
```

2）非递归版本

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []

        res = []
        stack = []
        node = root
        
        while stack or node:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                res.append(node.val)
                node = node.right
        return res
```

3）非递归版本。Morris中序遍历，空间复杂度$O(1)$。

==TODO==



###  145. 二叉树的后序遍历

> [145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/ "145. 二叉树的后序遍历")

遍历顺序：左子树，右子树，父节点。

1）递归版本

```python
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if root:
            left = self.postorderTraversal(root.left)
            res.extend(left)
            right = self.postorderTraversal(root.right)
            res.extend(right)
            res.append(root.val)
        return res
```

2）非递归版本。推荐方法。

```python
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        res =  []
        stack = []
        node = root
        visited = None  # 用来标记已访问的结点

        while stack or node:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack[-1]  
                # 关键：栈中最后一个结点，并检查该结点的右孩子是否为空或者刚被访问
                if node.right is None or node.right == visited: 
                    node = stack.pop()
                    res.append(node.val) # 访问该结点，并标记被访问
                    visited = node
                    node = None
                else:
                    node = node.right  # 在右子树进行一次后序遍历
        return res
```

3）非递归版本。该方法与前序遍历第 3 种方法类似。

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []

        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)    
        return res[::-1]
```



###  102. 二叉树的层序遍历

> [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/ "102. 二叉树的层序遍历")

一、题目

给你一个二叉树，请你返回其按 层序遍历 得到的节点值。（即逐层地，从左到右访问所有节点）。

二、解析

代码如下：

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        res = []
        queue = [root]
        while queue:
            cur_level_vals = []
            next_level_nodes = []
            for node in queue:
                cur_level_vals.append(node.val)
                if node.left: 
                    next_level_nodes.append(node.left)
                if node.right: 
                    next_level_nodes.append(node.right)
            res.append(cur_level_vals)
            queue = next_level_nodes
        return res
```



###  102-1. 二叉树的层序遍历II

返回一个列表。

1）不会往队列中添加空结点。

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        from collections import deque
        res = []
        queue = deque([root])  # 双向队列deque
        while queue:
            node = queue.popleft()
            res.append(node.val)
            if node.left: 
                queue.append(node.left)
            if node.right: 
                queue.append(node.right)
        return res
```

2）会往队列中添加空结点。

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root:
            return []

        from collections import deque
        res = []
        queue = deque([root])  # 双向队列deque
        while queue:
            node = queue.popleft()
            if node:
                res.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                pass  # 可以进行一些操作
        return res
```





###  103. 锯齿形层序遍历

> [103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/)

一、题目

给你二叉树的根节点 root ，返回其节点值的 锯齿形层序遍历 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

二、解析

代码如下：

```python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        res = []
        ordered = True
        queue = [root]
        while queue:
            cur_level_vals = []
            next_level_nodes = []
            for node in queue:
                cur_level_vals.append(node.val)
                if node.left: 
                    next_level_nodes.append(node.left)
                if node.right: 
                    next_level_nodes.append(node.right)
            if ordered:
                res.append(cur_level_vals)
            else:
                res.append(cur_level_vals[::-1])
            ordered = not ordered
            queue = next_level_nodes
        return res
```



###  235. 二叉搜索树的最近公共祖先

> [235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/ "235. 二叉搜索树的最近公共祖先")

一、题目

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

二、解析

代码如下：

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root
```



###  236. 二叉树的最近公共祖先

> [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/ "236. 二叉树的最近公共祖先")

一、题目

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

二、解析

代码如下：

```python
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if not root or root == p or root == q:
            return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left and not right:
            return None
        elif not left:
            return right
        elif not right:
            return left
        return root
```



###  257. 二叉树的所有路径

> [257. 二叉树的所有路径](https://leetcode-cn.com/problems/binary-tree-paths/ "257. 二叉树的所有路径")

一、题目

给定一个二叉树，返回所有从根节点到叶子节点的路径。

二、解析

利用层序遍历。访问到叶子结点时需要保存结果，也可以在 if 条件中添加判断条件。

代码如下：

```python
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if not root:
            return []
        
        from collections import deque
        res = []
        node_queue = deque([root])
        path_queue = deque([str(root.val)])
        while node_queue:
            node = node_queue.popleft()
            path = path_queue.popleft()
            if node.left:
                node_queue.append(node.left)
                path_queue.append(path + '->' + str(node.left.val))
            if node.right:
                node_queue.append(node.right)
                path_queue.append(path + '->' + str(node.right.val))
            if not node.left and not node.right:
                res.append(path)
        return res
```

###  112. 路经总和

> [112. 路径总和](https://leetcode-cn.com/problems/path-sum/ "112. 路径总和")

一、题目

给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。

说明: 叶子节点是指没有子节点的节点。

二、解析

1）使用模板

```python
class Solution:
    def hasPathSum(self, root: TreeNode, s: int) -> bool:
        if not root:
            return False
        
        from collections import deque
        res = []
        node_queue = deque([root])
        path_queue = deque([root.val])
        while node_queue:
            node = node_queue.popleft()
            path = path_queue.popleft()
            if node.left:
                node_queue.append(node.left)
                path_queue.append(path + node.left.val)
            if node.right:
                node_queue.append(node.right)
                path_queue.append(path + node.right.val)
            if not node.left and not node.right and path == s: # 叶子节点
                return True 
        return False
```

2）递归

```python
class Solution:
    def hasPathSum(self, root: TreeNode, s: int) -> bool:
        if not root:
            return False

        if not root.left and not root.right:  # if reach a leaf
            return s - root.val == 0
        return self.hasPathSum(root.left, s - root.val) or self.hasPathSum(root.right, s - root.val)
```



###  113. 路径总和 II

> [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/ "113. 路径总和 II")

一、题目

给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。

说明: 叶子节点是指没有子节点的节点。

二、解析

1）使用模板

```python
class Solution:
    def pathSum(self, root: TreeNode, s: int) -> List[List[int]]:
        if not root:
            return []

        from collections import deque
        res = []
        node_queue = deque([root])
        path_queue = deque([[root.val]])
        while node_queue:
            node = node_queue.popleft()
            path = path_queue.popleft()
            if node.left:
                node_queue.append(node.left)
                path_queue.append(path + [node.left.val])
            if node.right:
                node_queue.append(node.right)
                path_queue.append(path + [node.right.val])
            if not node.left and not node.right and sum(path) == s:
                res.append(path)
        
        return res
```

2、递归

```python
class Solution:
    def pathSum(self, root: TreeNode, s: int) -> List[List[int]]:
        def get_all_path(node):
            if not node:
                return []
            elif not node.left and not node.right:
                return[[node.val]]

            left = get_all_path(node.left)
            right = get_all_path(node.right)
            return [[node.val] + path for path in left + right]
        
        res = get_all_path(root)
        return [path for path in res if sum(path) == s]
```



###  437. 路径总和 III

> [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/ "437. 路径总和 III")

一、题目

给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。

路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

二、解析

代码如下：

```python
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> int:
        def helper(node, target):
            if not node:
                return [], 0  # path_sum, count
            left = helper(node.left, target)
            right = helper(node.right, target)
            path_sum = [ps + node.val for ps in left[0] + right[0]] + [node.val]
            count = left[1] + right[1]
            for ps in path_sum:
                if ps == target:
                    count += 1
            return path_sum, count
        
        return helper(root, target)[1]
```

递归函数不返回目标变量，而是直接在内部修改，递归更简洁。

```python
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> int:
        def helper(node, target):
            if not node:
                return []  # path_sum
            nonlocal res
            left = helper(node.left, target)
            right = helper(node.right, target)
            path_sum = [ps + node.val for ps in left + right] + [node.val]
            for ps in path_sum:
                if ps == target:
                    res += 1
            return path_sum
        
        res = 0
        helper(root, target)
        return res
```





###  105. 从前序与中序遍历序列构造二叉树

> [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/ "105. 从前序与中序遍历序列构造二叉树")

一、题目

根据一棵树的前序遍历与中序遍历构造二叉树。

注意：你可以假设树中没有重复的元素。

二、解析

代码如下：

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def helper(preorder, inorder):
            if not preorder:
                return None
            if len(preorder) == 1:
                return TreeNode(preorder[0])

            index = inorder.index(preorder[0])
            root = TreeNode(preorder[0])
            root.left = helper(preorder[1:index + 1], inorder[:index])
            root.right = helper(preorder[index + 1:], inorder[index + 1:])
            return root

        return helper(preorder, inorder)
```



###  106. 从中序与后序遍历序列构造二叉树

> [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/ "106. 从中序与后序遍历序列构造二叉树")

一、题目

根据一棵树的中序遍历与后序遍历构造二叉树。

注意:你可以假设树中没有重复的元素。

二、解析

代码如下：

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        def helper(inorder, postorder):
            if not inorder:
                return None
            elif len(inorder) == 1:
                return TreeNode(inorder[0])

            root = TreeNode(postorder[-1])
            index = inorder.index(postorder[-1])
            n_right = len(inorder) - index - 1
            root.left = helper(inorder[:index], postorder[:index])
            root.right = helper(inorder[index + 1:], postorder[-1 - n_right:-1])
            return root

        return helper(inorder, postorder)
```



###  889. 根据前序和后序遍历构造二叉树

> [889. 根据前序和后序遍历构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/ "889. 根据前序和后序遍历构造二叉树")

一、题目

返回与给定的前序和后序遍历匹配的任何二叉树。

pre 和 post 遍历中的值是不同的正整数。长度相等。

每个输入保证至少有一个答案。如果有多个答案，可以返回其中一个。

二、解析

代码如下：

```python
class Solution:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
        def helper(pre, post):
            if not pre or not post or len(pre) != len(post):
                return None
            
            root = TreeNode(pre[0])
            if len(pre) == 1:
                return root
            
            index = post.index(pre[1])
            count = index + 1
            
            root.left = helper(pre[1 : count + 1], post[:count])
            root.right = helper(pre[count + 1:], post[count:-1])
            return root

        return helper(pre, post)
```



###  1028. 从先序遍历还原二叉树

> [1028. 从先序遍历还原二叉树](https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/ "1028. 从先序遍历还原二叉树")

一、题目

我们从二叉树的根节点`root`开始进行深度优先搜索。

在遍历中的每个节点处，我们输出`D`条短划线（其中`D`是该节点的深度），然后输出该节点的值。（如果节点的深度为`D`，则其直接子节点的深度为`D + 1`。根节点的深度为`0`）。

如果节点只有一个子节点，那么保证该子节点为左子节点。

给出遍历输出`S`，还原树并返回其根节点`root`。

示例：

```纯文本
输入："1-2--3--4-5--6--7"

对应的二叉树：
         1
       /   \
      2     5
     / \   / \
    3   4 6   7
```

二、解析

1）栈模拟。

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/solution/cong-xian-xu-bian-li-huan-yuan-er-cha-shu-by-leetc/ "Leetcode官方题解")

代码如下：

```python
class Solution:
    def recoverFromPreorder(self, S: str) -> TreeNode:
        stack = []
        pos = 0
        while pos < len(S):
            level = 0
            while S[pos] == '-':
                level += 1
                pos += 1
            value = 0
            while pos < len(S) and S[pos].isdigit():
                value = value * 10 + (ord(S[pos]) - ord('0'))
                pos += 1
            node = TreeNode(value)
            if level == len(stack):
                if stack:
                    stack[-1].left = node
            else:
                stack = stack[:level]
                stack[-1].right = node
            stack.append(node)
        return stack[0]
```

2）用哈希表，更容易理解。

> 参考 [Leetcode-Actonmic](https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/comments/450316 "Leetcode-Actonmic")

代码如下：

```python
class Solution:
    def recoverFromPreorder(self, S: str) -> TreeNode:
        pos = 0
        saved = {0: None}
        while pos < len(S):
            level = 0
            while S[pos] == '-':
                level += 1
                pos += 1
            value = 0
            while pos < len(S) and S[pos].isdigit():
                value = value * 10 + (ord(S[pos]) - ord('0'))
                pos += 1
            node = TreeNode(value)
            if level == 0:
                saved[0] = node
            else:
                parent = saved[level - 1]
                if parent.left is None:
                    parent.left = node
                else:
                    parent.right = node
                saved[level] = node
        return saved[0]
```



###  1008. 先序遍历构造二叉搜索树

> [1008. 先序遍历构造二叉树](https://leetcode-cn.com/problems/construct-binary-search-tree-from-preorder-traversal/comments/ "1008. 先序遍历构造二叉树")

一、题目

返回与给定先序遍历 preorder 相匹配的二叉搜索树（binary search tree）的根结点。

二、解析

根据先序遍历构造二叉搜索树。

代码如下：

```python
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        def helper(preorder):
            if not preorder:
                return None
            if len(preorder) == 1:
                return TreeNode(preorder[0])

            root = TreeNode(preorder[0])
            begin, end = 1, 1
            while end < len(preorder) and preorder[end] < preorder[0]:
                end += 1
            
            root.left = helper(preorder[begin:end])
            root.right = helper(preorder[end:])
            return root
        
        return helper(preorder)
```



###  104. 二叉树的最大深度

> [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

一、题目

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例：给定二叉树 \[3,9,20,null,null,15,7]，

```python
    3
   / \
  9  20
    /  \
   15   7
```

返回它的最大深度 3 。

二、解析

代码如下：

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        def helper(node):
            if not node:
                return 0
            left = helper(node.left)
            right = helper(node.right)
            return max(left, right) + 1
        return helper(root)
```



###  110. 平衡二叉树

> [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/ "110. 平衡二叉树")

一、题目

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。

二、解析

代码如下：

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def helper(node):
            if not node:
                return 0, True  # depth, is_balance
            left = helper(node.left)
            right = helper(node.right)
            depth = max(left[0], right[0]) + 1
            is_balance = left[1] and right[1] and abs(left[0] - right[0]) <= 1
            return depth, is_balance
        return helper(root)[1]
```



###  543. 二叉树中的直径

> [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/ "543. 二叉树的直径")

一、题目

给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。

二、解析

代码如下：

```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        def helper(node):
            if not node:
                return 0, 0  # depth, diameter
            left = helper(node.left)
            right = helper(node.right)
            depth = max(left[0], right[0]) + 1
            diameter = max(left[0] + right[0], left[1], right[1])
            return depth, diameter
        return helper(root)[1]
```



###  536. 二叉树的坡度

> [563. 二叉树的坡度](https://leetcode-cn.com/problems/binary-tree-tilt/ "563. 二叉树的坡度")

一、题目

给定一个二叉树，计算整个树的坡度。

一个树的节点的坡度定义即为，该节点左子树的结点之和和右子树结点之和的差的绝对值。空结点的的坡度是0。

整个树的坡度就是其所有节点的坡度之和。

二、解析

代码如下：

```python
class Solution:
    def findTilt(self, root: TreeNode) -> int:
        def helper(node):
            if not node:
                return 0, 0 # sum, tilt
            left = helper(node.left)
            right = helper(node.right)
            sum_ = left[0] + right[0] + node.val
            tilt = left[1] + right[1] + abs(left[0] - right[0])
            return sum_, tilt  
        return helper(root)[1]
```



###  124. 二叉树中的最大路径和

> [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/ "124. 二叉树中的最大路径和")

一、题目

给定一个非空二叉树，返回其最大路径和。

本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径至少包含一个节点，且不一定经过根节点。

二、解析

如果当前结点的值小于0，那么路径可以不包含当前结点，当前路径的和最小为0。

代码如下：

```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        def helper(node):
            if not node:
                return float("-inf"), 0  # res, the larger sum
            left = helper(node.left)
            right = helper(node.right)
            left_sum = max(0, left[1])
            right_sum = max(0, right[1])
            res = max(left[0], right[0], left_sum + node.val + right_sum)
            larger_sum = max(left_sum, right_sum) + node.val
            return res, larger_sum

        return helper(root)[0]
```

递归函数不返回目标变量，而是直接在内部修改，递归更简洁。

```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        def helper(node):
            if not node:
                return 0
            nonlocal res
            left = helper(node.left)
            right = helper(node.right)
            left_sum = max(0, left)
            right_sum = max(0, right)
            res = max(res, left_sum + node.val + right_sum)
            return max(left_sum, right_sum) + node.val
        
        res = float('-inf')
        maxPath(root)
        return res
```



###  437. 路径总和 III

> [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/ "437. 路径总和 III")

一、题目

给定一个二叉树，它的每个结点都存放着一个整数值。

找出路径和等于给定数值的路径总数。

路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

二叉树不超过1000个节点，且节点数值范围是 \[-1000000,1000000] 的整数。

二、解析

代码如下：

```python
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> int:
        def helper(node, target):
            if not node:
                return [], 0  # path_sum, count
            left = helper(node.left, target)
            right = helper(node.right, target)
            path_sum = [ps + node.val for ps in left[0] + right[0]] + [node.val]
            count = left[1] + right[1]
            for ps in path_sum:
                if ps == target:
                    count += 1
            return path_sum, count
        
        return helper(root, target)[1]
```

递归函数不返回目标变量，而是直接在内部修改，递归更简洁。

```python
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> int:
        def helper(node, target):
            if not node:
                return []  # path_sum
            nonlocal res
            left = helper(node.left, target)
            right = helper(node.right, target)
            path_sum = [ps + node.val for ps in left + right] + [node.val]
            for ps in path_sum:
                if ps == target:
                    res += 1
            return path_sum
        
        res = 0
        helper(root, target)
        return res
```



###  226. 翻转二叉树

> [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/ "226. 翻转二叉树")

一、题目

翻转一棵二叉树。

二、解析

代码如下：

```python
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None

        from collections import deque
        queue = deque([root])
        while queue:
            node = queue.popleft()
            node.left, node.right = node.right, node.left
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return root
```



###  297. 二叉树的序列化与反序列化

> [297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/ "297. 二叉树的序列化与反序列化")

一、题目

序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

二、解析

层序遍历。

代码如下：

```python
class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string."""
        if not root:
            return "[]"
        
        from collections import deque
        queue = deque([root])
        values = []
        while queue:
            node = queue.popleft()
            if node:
                values.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                values.append("null")
        
        return "[" + ",".join(values) + "]"

    def deserialize(self, data):
        """Decodes your encoded data to tree."""
        if data == "[]":
            return None

        values = data[1:-1].split(",")
        i = 0
        root = TreeNode(int(values[i]))
        queue = deque([root])

        while queue:
            node = queue.popleft()
            i += 1
            if values[i] != "null":
                node.left = TreeNode(int(values[i]))
                queue.append(node.left)
            i += 1
            if values[i] != "null":
                node.right = TreeNode(int(values[i]))
                queue.append(node.right)

        return root
```



###  331. 验证二叉树的前序序列化

> [331. 验证二叉树的前序序列化](https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/ "331. 验证二叉树的前序序列化")

一、题目

序列化二叉树的一种方法是使用前序遍历。当我们遇到一个非空节点时，我们可以记录下这个节点的值。如果它是一个空节点，我们可以使用一个标记值记录，例如`#`。

例如，上面的二叉树可以被序列化为字符串"9,3,4,#,#,1,#,#,2,#,6,#,#"，其中 # 代表一个空节点。

给定一串以逗号分隔的序列，验证它是否是正确的二叉树的前序序列化。编写一个在不重构树的条件下的可行算法。

每个以逗号分隔的字符或为一个整数或为一个表示`null`指针的`'#'`。

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/solution/yan-zheng-er-cha-shu-de-qian-xu-xu-lie-hua-by-leet/ "Leetcode官方题解")

栈模拟。代码如下：

```python
class Solution:
    def isValidSerialization(self, preorder: str) -> bool:
        slots = 1
        for node in preorder.split(','):
            slots -= 1
            if slots < 0:
                return False
            if node != '#':
                slots += 2
        
        return slots == 0
```



###  606. 根据二叉树创建字符串

> [606. 根据二叉树创建字符串](https://leetcode-cn.com/problems/construct-string-from-binary-tree/ "606. 根据二叉树创建字符串")

一、题目

你需要采用前序遍历的方式，将一个二叉树转换成一个由括号和整数组成的字符串。

空节点则用一对空括号 "()" 表示。而且你需要省略所有不影响字符串与原始二叉树之间的一对一映射关系的空括号对。

二、解析

递归比较简单。

```python
class Solution:
    def tree2str(self, t: TreeNode) -> str:
        if not t:
            return ''
        elif not t.left and not t.right:
            return str(t.val)
        elif not t.left:
            return str(t.val) + '()(' + self.tree2str(t.right) + ')'
        elif not t.right:
            return str(t.val) + '(' + self.tree2str(t.left) + ')'
        return str(t.val) + '(' + self.tree2str(t.left) + ')(' + self.tree2str(t.right) + ')'
```



###  536. 根据字符串创建二叉树

> [536. 从字符串生成二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-string/ "536. 从字符串生成二叉树")[Lintcode 880. 字符串构造二叉树](https://www.lintcode.com/problem/construct-binary-tree-from-string "Lintcode 880. 字符串构造二叉树")

一、题目

你需要根据一个由括号和整数组成的字符串中构造一颗二叉树。

输入的整个字符串表示一个二叉树。它包含一个整数，以及其后跟随的0\~2对括号。该整数表示根的值，而一对括号里的字符串代表一个具有相同结构的子二叉树。

示例：

```纯文本
输入: "-4(2(3)(1))(6(5))"
输入字符串对应的二叉树：
      -4
     /   \
    2     6
   / \   / 
  3   1 5 
```

**注意**：Leetcode需要会员，Lintcode不需要。

二、解析

迭代法和递归法。

1）迭代。

> [536. Construct Binary Tree from String 从带括号字符串构建二叉树](https://segmentfault.com/a/1190000016808160 "536. Construct Binary Tree from String 从带括号字符串构建二叉树")

代码如下：

```python
class Solution:
    def str2tree(self, s):
        if not s:
            return None
            
        stack = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch.isdigit() or ch == '-':
                begin = i
                while i + 1 < len(s) and s[i + 1].isdigit():
                    i += 1
                node = TreeNode(int(s[begin:i + 1]))
                if stack:
                    parent = stack[-1]
                    if parent.left is None:
                        parent.left = node
                    else:
                        parent.right = node
                stack.append(node)
            elif ch == ')':
                stack.pop()
            i += 1
        return stack[0]  
```

2）递归

> [Lintcode-xiongxiong](https://www.lintcode.com/problem/construct-binary-tree-from-string/note/187028 "Lintcode-xiongxiong")

代码如下：

```python
class Solution:
    def str2tree(self, s):
        if not s:
            return None
        left_begin = s.find('(')
        if left_begin == -1:
            return TreeNode(int(s))

        left_end = left_begin + 1
        num = 1
        while left_end < len(s):
            if s[left_end] == '(':
                num += 1
            elif s[left_end] == ')':
                num -= 1
                if num == 0:
                    break
            left_end += 1

        root = TreeNode(int(s[:left_begin]))
        root.left = self.str2tree(s[left_begin + 1: left_end])
        right_end = s.rfind(')')
        if right_end > left_end + 1:
            root.right = self.str2tree(s[left_end + 2: right_end])

        return root
```



### 426. 将二叉搜索树转化为排序的双向链表

> [426. 将二叉搜索树转化为排序的双向链表](https://leetcode-cn.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/ "426. 将二叉搜索树转化为排序的双向链表")

一、题目

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

二、解析

中序遍历，修改一下指针。

代码如下：

```python
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if root is None:
            return root

        pre = head = Node(None)
        stack = []
        node = root
        while node or stack:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                node.left = pre
                pre.right = node
                pre = node
                node = node.right

        pre.right = head.right
        head.right.left = pre
        return head.right
```



### 109. 有序链表转换二叉搜索树

> [109. 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/ "109. 有序链表转换二叉搜索树")

一、题目

给定一个单链表，其中的元素按升序排序，将其转换为**高度平衡的二叉搜索树**。

本题中，一个高度平衡二叉树是指一个二叉树每个节点的左右两个子树的高度差的绝对值不超过 1。

示例：

```纯文本
给定的有序链表： [-10, -3, 0, 5, 9],

一个可能的答案是：[0, -3, 9, -10, null, 5], 它可以表示下面这个高度平衡二叉搜索树：

      0
     / \
   -3   9
   /   /
 -10  5
```

二、解析

快慢指针 + 递归。

代码如下：

```python
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        if not head:
            return None
        elif not head.next:
            return TreeNode(head.val)
        slow = fast = head
        pre = None
        while fast and fast.next:
            pre = slow
            slow = slow.next
            fast = fast.next.next
        pre.next = None
        root = TreeNode(slow.val)
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(slow.next)
        return root
```



### 669. 修剪二叉搜索树

> [669. 修剪二叉搜索树](https://leetcode-cn.com/problems/trim-a-binary-search-tree/ "669. 修剪二叉搜索树")

一、题目

给定一个二叉搜索树，同时给定最小边界L和最大边界R。通过修剪二叉搜索树，使得所有节点的值在\[L, R]中 (R>=L) 。你可能需要改变树的根节点，所以结果应当返回修剪好的二叉搜索树的新的根节点。

二、解析

代码如下：

```python
class Solution:
    def trimBST(self, root: TreeNode, L: int, R: int) -> TreeNode:
        if not root:
            return None
        elif root.val < L:
            return self.trimBST(root.right, L, R)
        elif root.val > R:
            return self.trimBST(root.left, L, R)

        root.left = self.trimBST(root.left, L, R)
        root.right = self.trimBST(root.right, L, R)
        return root
```



### 538. 把二叉搜索树转换为累加树

> [538. 把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/ "538. 把二叉搜索树转换为累加树")

一、题目

给定一个二叉搜索树（Binary Search Tree），把它转换成为累加树（Greater Tree)，使得每个节点的值是原来的节点值加上所有大于它的节点值之和。

二、解析

代码如下：

```python
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        def convert(node, num):
            if not node:
                return num
            node.val += convert(node.right, num)
            return convert(node.left, node.val)
             
        convert(root, 0)
        return root
```



### 450. 删除二叉搜索树中的节点

> [450. 删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst/ "450. 删除二叉搜索树中的节点")

一、题目

给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：首先找到需要删除的节点；如果找到了，删除它。

说明： 要求算法时间复杂度为 O(h)，h 为树的高度。

二、解析

1）递归。

TODO

2）迭代。

代码如下：

```python
class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        def delete_node_by_copy(left_child, node):
            child = left_child
            parent = node
            while child.right:
                parent = child
                child = child.right
            if child == left_child:
                node.val = child.val
                parent.left = child.left
            else:
                node.val = child.val
                parent.right = child.left
        
        parent = None
        cur = root
        while cur and cur.val != key:
            parent = cur
            cur = cur.left if cur.val > key else cur.right
        
        if cur == None:
            return root
        if cur is root:
            if cur.left is None and cur.right is None:
                return None
            if cur.left and cur.right:
                delete_node_by_copy(cur.left, cur)
                return root
            if cur.left:
                return cur.left
            if cur.right:
                return cur.right
        if cur.left is None or cur.right is None:
            child = cur.left if cur.left else cur.right
            if parent.left is cur:
                parent.left = child
            else:
                parent.right = child
            return root
        
        delete_node_by_copy(cur.left, cur)
        return root
```



### 173. 二叉搜索树迭代器

> [173. 二叉搜索树迭代器](https://leetcode-cn.com/problems/binary-search-tree-iterator/ "173. 二叉搜索树迭代器")

一、题目

实现一个二叉搜索树迭代器。你将使用二叉搜索树的根节点初始化迭代器。

调用`next()`将返回二叉搜索树中的下一个最小的数。

二、解析

利用中序遍历。

```python
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.stack = []
        self._inorder(root)
    
    def _inorder(self, node):
        cur = node
        while cur:
            self.stack.append(cur)
            cur = cur.left 

    def next(self) -> int:
        """
        @return the next smallest number
        """
        cur = self.stack.pop()
        val = cur.val
        cur = cur.right
        self._inorder(cur)
        return val

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return len(self.stack) > 0
```



### 602. 二叉树最大宽度

> [662. 二叉树最大宽度](https://leetcode-cn.com/problems/maximum-width-of-binary-tree/ "662. 二叉树最大宽度")

一、题目

给定一个二叉树，编写一个函数来获取这个树的最大宽度。树的宽度是所有层中的最大宽度。这个二叉树与满二叉树（full binary tree）结构相同，但一些节点为空。

每一层的宽度被定义为两个端点（该层最左和最右的非空节点，两端点间的null节点也计入长度）之间的长度。

示例1

```text
输入：root = [1,3,2,5,3,null,9]
输出：4
解释：最大宽度出现在树的第 3 层，宽度为 4 (5,3,null,9) 。
```

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/maximum-width-of-binary-tree/solution/er-cha-shu-zui-da-kuan-du-by-leetcode/ "Leetcode官方题解")

广度优先搜索或深度优先搜索。

1）广度优先搜索

```python
class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        queue = [(root, 0, 0)]
        res = 0
        cur_depth = left = 0
        while queue:
            node, depth, pos = queue.pop(0)
            if node:
                queue.append((node.left, depth + 1, pos * 2))
                queue.append((node.right, depth + 1, pos * 2 + 1))
                if cur_depth != depth:
                    cur_depth = depth
                    left = pos
                res = max(pos - left + 1, res)
        return res
```

2）深度优先搜索

```python
class Solution(object):
    def widthOfBinaryTree(self, root):
        self.ans = 0
        left = {}
        def dfs(node, depth = 0, pos = 0):
            if node:
                left.setdefault(depth, pos)
                self.ans = max(self.ans, pos - left[depth] + 1)
                dfs(node.left, depth + 1, pos * 2)
                dfs(node.right, depth + 1, pos * 2 + 1)

        dfs(root)
        return self.ans
```



### 222. 完全二叉树的结点个数

> [222. 完全二叉树的结点个数](https://leetcode-cn.com/problems/count-complete-tree-nodes/ "222. 完全二叉树的结点个数")

一、题目

给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。

完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1\~ 2h 个节点。

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/count-complete-tree-nodes/solution/wan-quan-er-cha-shu-de-jie-dian-ge-shu-by-leetcode/ "Leetcode官方题解")

遍历一次，或者使用二分搜索。

1）遍历一次，递归

```python
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        return 1 + self.countNodes(root.right) + self.countNodes(root.left) if root else 0
```

2）二分搜索

```python
class Solution:
    def compute_depth(self, node: TreeNode) -> int:
        d = 0
        while node.left:
            node = node.left
            d += 1
        return d

    def exists(self, idx: int, d: int, node: TreeNode) -> bool:
        left, right = 0, 2**d - 1
        for _ in range(d):
            mid = left + (right - left) // 2
            if idx > mid:
                node = node.right
                left = mid + 1
            else:
                node = node.left
                right = mid
        return node is not None
        
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        d = self.compute_depth(root)
        if d == 0:
            return 1
        
        left, right = 0, 2**d
        while left < right:
            mid = left + (right - left) // 2
            if self.exists(mid, d, root):
                left = mid + 1
            else:
                right = mid

        return (2**d - 1) + left
```



### 958. 二叉树的完全性检验

> [958. 二叉树的完全性检验](https://leetcode.cn/problems/check-completeness-of-a-binary-tree/)

一、题目

给定一个二叉树的 root ，确定它是否是一个 完全二叉树 。

在一个 完全二叉树 中，除了最后一个关卡外，所有关卡都是完全被填满的，并且最后一个关卡中的所有节点都是尽可能靠左的。它可以包含 1 到 2h 节点之间的最后一级 h 。

示例 1：

```text
输入：root = [1,2,3,4,5,6]
输出：true
解释：最后一层前的每一层都是满的（即，结点值为 {1} 和 {2,3} 的两层），且最后一层中的所有结点（{4,5,6}）都尽可能地向左。

```

二、解析

使用层序遍历【会往队列中添加空结点】。如果出现过空结点，就不能再有控结点。

代码如下：

```python
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True

        from collections import deque
        queue = deque([root])  # 双向队列deque
        none_node_flag = False
        while queue:
            node = queue.popleft()
            if node:
                if none_node_flag: return False
                queue.append(node.left)
                queue.append(node.right)
            else:
                none_node_flag = True
        return True
```



### 1325. 删除给定值的叶子节点

> [1325. 删除给定值的叶子节点](https://leetcode-cn.com/problems/delete-leaves-with-a-given-value/ "1325. 删除给定值的叶子节点")

一、题目

给你一棵以`root`为根的二叉树和一个整数`target`，请你删除所有值为`target`的叶子节点。

注意，一旦删除值为`target`的叶子节点，它的父节点就可能变成叶子节点；如果新叶子节点的值恰好也是`target`，那么这个节点也应该被删除。

也就是说，你需要重复此过程直到不能继续删除。

二、解析

> 参考 [Leetcode官方题解](https://leetcode-cn.com/problems/delete-leaves-with-a-given-value/solution/shan-chu-gei-ding-zhi-de-xie-zi-jie-dian-by-leet-2/ "Leetcode官方题解")

代码如下：

```python
class Solution:
    def removeLeafNodes(self, root: TreeNode, target: int) -> TreeNode:
        if not root:
            return None
        root.left = self.removeLeafNodes(root.left, target)
        root.right = self.removeLeafNodes(root.right, target)
        if not root.left and not root.right and root.val == target:
            return None
        return root
```



### 99. 恢复二叉搜索树

> [99. 恢复二叉搜索树](https://leetcode-cn.com/problems/recover-binary-search-tree/ "99. 恢复二叉搜索树")

一、题目

二叉搜索树中的两个节点被错误地交换。

请在不改变其结构的情况下，恢复这棵树。

二、解析

代码如下：

```python
class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return
        stack = []
        node = root
        all_nodes = []
        while stack or node:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                all_nodes.append(node)
                node = node.right
        
        wrong_nodes = []
        for i in range(1, len(all_nodes)):
            if all_nodes[i - 1].val > all_nodes[i].val:
                wrong_nodes.append(all_nodes[i -1])
                wrong_nodes.append(all_nodes[i])

        if len(wrong_nodes) == 2:
            n1, n2 = wrong_nodes
            n1.val, n2.val = n2.val, n1.val
        elif len(wrong_nodes) == 4:
            n1, n2 = wrong_nodes[0], wrong_nodes[3]
            n1.val, n2.val = n2.val, n1.val
```



### LCR 143. 树的子结构

> [LCR 143. 子结构判断](https://leetcode.cn/problems/shu-de-zi-jie-gou-lcof/)

一、题目

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:

```text
给定的树 A:
     3
    / \
   4   5
  / \
 1   2
给定的树 B：
   4 
  /
 1

返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

```

二、解析

使用层序遍历。每个结点都判断一次是否是子结构。

代码如下：

```python
class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def check(root1, root2):
            if not root1 and not root2:
                return True
            elif root1 and not root2:
                return True
            elif not root1 and root2:
                return False
            return root1.val == root2.val and check(root1.left, root2.left) and check(root1.right, root2.right)

        if not B:
            return False
        
        from collections import deque
        queue = deque([A])
        while queue:
            node = queue.popleft()
            if check(node, B):
                return True
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return False
```



## 其他

###  使用层序遍历构造二叉树

一、题目

根据层序遍历构造二叉树。

举例：

```纯文本
数组为：["1","2","3","null","null","4","5","null","null","null","null"]

对应的二叉树：
        1
       / \
      2   3
         / \
        4   5
```

二、解析

1）迭代

```python
def create_binary_tree(data):
    if not data:
        return None
        
    i = 0
    root = TreeNode(int(data[i]))
    queue = deque([root])
    while queue:
        node = queue.popleft()
        i += 1
        if data[i] != "null":
            node.left = TreeNode(int(data[i]))
            queue.append(node.left)
        i += 1
        if data[i] != "null":
            node.right = TreeNode(int(data[i]))
            queue.append(node.right)

    return root
```

