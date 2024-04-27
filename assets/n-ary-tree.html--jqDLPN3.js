import{_ as t}from"./plugin-vue_export-helper-x3n3nnut.js";import{r as o,o as c,c as l,d as i,a as n,e,b as p,f as s}from"./app-wnqBpBOb.js";const u={},r=s(`<h2 id="基础" tabindex="-1"><a class="header-anchor" href="#基础"><span>基础</span></a></h2><h3 id="树节点" tabindex="-1"><a class="header-anchor" href="#树节点"><span>树节点</span></a></h3><p>1）Python</p><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="language-python"><code><span class="token keyword">class</span> <span class="token class-name">Node</span><span class="token punctuation">:</span>
    <span class="token keyword">def</span> <span class="token function">__init__</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> val<span class="token operator">=</span><span class="token boolean">None</span><span class="token punctuation">,</span> children<span class="token operator">=</span><span class="token boolean">None</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        self<span class="token punctuation">.</span>val <span class="token operator">=</span> val
        self<span class="token punctuation">.</span>children <span class="token operator">=</span> children  <span class="token comment"># chilren is a list</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="leetcode-编程题" tabindex="-1"><a class="header-anchor" href="#leetcode-编程题"><span>Leetcode 编程题</span></a></h2><h3 id="_589-n叉树的前序遍历" tabindex="-1"><a class="header-anchor" href="#_589-n叉树的前序遍历"><span>589. N叉树的前序遍历</span></a></h3>`,6),d={href:"https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/",title:"589. N叉树的前序遍历",target:"_blank",rel:"noopener noreferrer"},k=s(`<p>一、题目</p><p>给定一个 N 叉树，返回其节点值的前序遍历。</p><p>二、解析</p><p>1）递归版本</p><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="language-python"><code><span class="token keyword">class</span> <span class="token class-name">Solution</span><span class="token punctuation">:</span>
    <span class="token keyword">def</span> <span class="token function">preorder</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> root<span class="token punctuation">:</span> <span class="token string">&#39;Node&#39;</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> List<span class="token punctuation">[</span><span class="token builtin">int</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
        res <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
        <span class="token keyword">if</span> root<span class="token punctuation">:</span>
            res<span class="token punctuation">.</span>append<span class="token punctuation">(</span>root<span class="token punctuation">.</span>val<span class="token punctuation">)</span>
            <span class="token keyword">for</span> c <span class="token keyword">in</span> root<span class="token punctuation">.</span>children<span class="token punctuation">:</span>
                vals <span class="token operator">=</span> self<span class="token punctuation">.</span>preorder<span class="token punctuation">(</span>c<span class="token punctuation">)</span>
                res<span class="token punctuation">.</span>extend<span class="token punctuation">(</span>vals<span class="token punctuation">)</span>

        <span class="token keyword">return</span> res
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>2）非递归版本</p><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="language-python"><code><span class="token keyword">class</span> <span class="token class-name">Solution</span><span class="token punctuation">:</span>
    <span class="token keyword">def</span> <span class="token function">preorder</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> root<span class="token punctuation">:</span> <span class="token string">&#39;Node&#39;</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> List<span class="token punctuation">[</span><span class="token builtin">int</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
        <span class="token keyword">if</span> <span class="token keyword">not</span> root<span class="token punctuation">:</span>
            <span class="token keyword">return</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

        res <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
        stack <span class="token operator">=</span> <span class="token punctuation">[</span>root<span class="token punctuation">]</span>
        <span class="token keyword">while</span> stack<span class="token punctuation">:</span>
            node <span class="token operator">=</span> stack<span class="token punctuation">.</span>pop<span class="token punctuation">(</span><span class="token punctuation">)</span>
            res<span class="token punctuation">.</span>append<span class="token punctuation">(</span>node<span class="token punctuation">.</span>val<span class="token punctuation">)</span>
            stack<span class="token punctuation">.</span>extend<span class="token punctuation">(</span>node<span class="token punctuation">.</span>children<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">:</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
        <span class="token keyword">return</span> res
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_590-n叉树的后序遍历" tabindex="-1"><a class="header-anchor" href="#_590-n叉树的后序遍历"><span>590. N叉树的后序遍历</span></a></h3>`,8),v={href:"https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/",title:"590. N叉树的后序遍历",target:"_blank",rel:"noopener noreferrer"},m=s(`<p>一、题目</p><p>给定一个 N 叉树，返回其节点值的后序遍历。</p><p>二、解析</p><p>1）递归</p><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="language-python"><code><span class="token keyword">class</span> <span class="token class-name">Solution</span><span class="token punctuation">:</span>
    <span class="token keyword">def</span> <span class="token function">postorder</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> root<span class="token punctuation">:</span> <span class="token string">&#39;Node&#39;</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> List<span class="token punctuation">[</span><span class="token builtin">int</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
        res <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
        <span class="token keyword">if</span> root<span class="token punctuation">:</span>
            <span class="token keyword">for</span> c <span class="token keyword">in</span> root<span class="token punctuation">.</span>children<span class="token punctuation">:</span>
                vals <span class="token operator">=</span> self<span class="token punctuation">.</span>postorder<span class="token punctuation">(</span>c<span class="token punctuation">)</span>
                res<span class="token punctuation">.</span>extend<span class="token punctuation">(</span>vals<span class="token punctuation">)</span>
            res<span class="token punctuation">.</span>append<span class="token punctuation">(</span>root<span class="token punctuation">.</span>val<span class="token punctuation">)</span>
        
        <span class="token keyword">return</span> res
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>2）非递归</p><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="language-python"><code><span class="token keyword">class</span> <span class="token class-name">Solution</span><span class="token punctuation">:</span>
    <span class="token keyword">def</span> <span class="token function">postorder</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> root<span class="token punctuation">:</span> <span class="token string">&#39;Node&#39;</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> List<span class="token punctuation">[</span><span class="token builtin">int</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
        <span class="token keyword">if</span> root <span class="token keyword">is</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
            <span class="token keyword">return</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
        
        res <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
        stack <span class="token operator">=</span> <span class="token punctuation">[</span>root<span class="token punctuation">]</span>
        <span class="token keyword">while</span> stack<span class="token punctuation">:</span>
            node <span class="token operator">=</span> stack<span class="token punctuation">.</span>pop<span class="token punctuation">(</span><span class="token punctuation">)</span>
            res<span class="token punctuation">.</span>append<span class="token punctuation">(</span>node<span class="token punctuation">.</span>val<span class="token punctuation">)</span>
            stack<span class="token punctuation">.</span>extend<span class="token punctuation">(</span>node<span class="token punctuation">.</span>children<span class="token punctuation">)</span>
                
        <span class="token keyword">return</span> res<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">:</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_429-n叉树的层序遍历" tabindex="-1"><a class="header-anchor" href="#_429-n叉树的层序遍历"><span>429. N叉树的层序遍历</span></a></h3>`,8),b={href:"https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/",title:"429. N叉树的层序遍历",target:"_blank",rel:"noopener noreferrer"},h=s(`<p>一、题目</p><p>给定一个 N 叉树，返回其节点值的<em>层序遍历</em>。（即从左到右，逐层遍历）。</p><p>树的序列化输入是用层序遍历，每组子节点都由 null 值分隔（参见示例）。</p><p>二、解析</p><p>代码如下：</p><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="language-python"><code><span class="token keyword">class</span> <span class="token class-name">Solution</span><span class="token punctuation">:</span>
    <span class="token keyword">def</span> <span class="token function">levelOrder</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> root<span class="token punctuation">:</span> <span class="token string">&#39;Node&#39;</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> List<span class="token punctuation">[</span>List<span class="token punctuation">[</span><span class="token builtin">int</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
        <span class="token keyword">if</span> <span class="token keyword">not</span> root<span class="token punctuation">:</span>
            <span class="token keyword">return</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
        
        queue <span class="token operator">=</span> <span class="token punctuation">[</span>root<span class="token punctuation">]</span>
        res <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
        <span class="token keyword">while</span> queue<span class="token punctuation">:</span>
            cur_level_vals <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
            next_level_nodes <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
            <span class="token keyword">for</span> node <span class="token keyword">in</span> queue<span class="token punctuation">:</span>
                cur_level_vals<span class="token punctuation">.</span>append<span class="token punctuation">(</span>node<span class="token punctuation">.</span>val<span class="token punctuation">)</span>
                <span class="token keyword">if</span> node<span class="token punctuation">.</span>children<span class="token punctuation">:</span>
                    next_level_nodes<span class="token punctuation">.</span>extend<span class="token punctuation">(</span>node<span class="token punctuation">.</span>children<span class="token punctuation">)</span>
            res<span class="token punctuation">.</span>append<span class="token punctuation">(</span>cur_level_vals<span class="token punctuation">)</span>
            queue <span class="token operator">=</span> next_level_nodes
        <span class="token keyword">return</span> res
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_429-1-n叉树的层序遍历" tabindex="-1"><a class="header-anchor" href="#_429-1-n叉树的层序遍历"><span>429-1. N叉树的层序遍历</span></a></h3><p>层序遍历，返回一个列表。</p><p>代码如下：</p><div class="language-python line-numbers-mode" data-ext="py" data-title="py"><pre class="language-python"><code><span class="token keyword">class</span> <span class="token class-name">Solution</span><span class="token punctuation">:</span>
    <span class="token keyword">def</span> <span class="token function">levelOrder</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> root<span class="token punctuation">:</span> <span class="token string">&#39;Node&#39;</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> List<span class="token punctuation">[</span>List<span class="token punctuation">[</span><span class="token builtin">int</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
        <span class="token keyword">if</span> <span class="token keyword">not</span> root<span class="token punctuation">:</span>
            <span class="token keyword">return</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
        
        <span class="token keyword">from</span> collections <span class="token keyword">import</span> deque
        res <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
        queue <span class="token operator">=</span> deque<span class="token punctuation">(</span><span class="token punctuation">[</span>root<span class="token punctuation">]</span><span class="token punctuation">)</span>  <span class="token comment"># 双向队列deque</span>
        <span class="token keyword">while</span> queue<span class="token punctuation">:</span>
            node <span class="token operator">=</span> queue<span class="token punctuation">.</span>popleft<span class="token punctuation">(</span><span class="token punctuation">)</span>
            res<span class="token punctuation">.</span>append<span class="token punctuation">(</span>node<span class="token punctuation">.</span>val<span class="token punctuation">)</span>
            queue<span class="token punctuation">.</span>extend<span class="token punctuation">(</span>node<span class="token punctuation">.</span>children<span class="token punctuation">)</span>
        <span class="token keyword">return</span> res
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,10);function y(f,_){const a=o("ExternalLinkIcon");return c(),l("div",null,[i(" more "),r,n("blockquote",null,[n("p",null,[n("a",d,[e("589. N叉树的前序遍历"),p(a)])])]),k,n("blockquote",null,[n("p",null,[n("a",v,[e("590. N叉树的后序遍历"),p(a)])])]),m,n("blockquote",null,[n("p",null,[n("a",b,[e("429. N叉树的层序遍历"),p(a)])])]),h])}const x=t(u,[["render",y],["__file","n-ary-tree.html.vue"]]);export{x as default};