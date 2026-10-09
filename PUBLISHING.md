# 发布到 GitHub 的完整步骤

本地仓库**已经初始化并完成首次提交**（62 个文件、`.git` 868 KB）。
你只需要做下面 5 步。每一步都有"验证"命令，出问题能立刻定位。

---

## 0. 先填身份（首次提交用的是占位身份）

```bash
cd /home/liqingchen/project/cbe-release
git config user.name  "你的名字拼音"
git config user.email "你的邮箱@example.com"
git commit --amend --reset-author --no-edit     # 把首次提交的作者改成你
git log -1 --format='%an <%ae>'                 # 验证
```

> 这只有本地生效（不带 `--global`）。如果你希望所有仓库都用这个身份，加 `--global`。

---

## 1. 在 GitHub 上建空仓库

浏览器打开 https://github.com/new ，填：

| 字段 | 值 |
|---|---|
| Repository name | `cbe` |
| Description | Coverage-based extrapolation detection for machine-learned interatomic potentials |
| Visibility | Public（要发 Zenodo DOI 必须 public） |
| **Initialize this repository with** | **全部不要勾**（不要 README / .gitignore / license —— 我们本地都有了） |

创建后页面会给你一个地址，形如 `https://github.com/<你的用户名>/cbe.git`。

### 认证：二选一

**A. HTTPS + Personal Access Token（不用装东西，推荐先试这个）**

1. https://github.com/settings/tokens → *Generate new token (classic)*
2. 勾 **`repo`** 权限，有效期选 90 天，生成后**复制**（只显示一次）
3. push 时用户名填你的 GitHub 用户名，**密码栏粘贴 token**

**B. SSH key（一劳永逸，但要配一次）**

```bash
ssh-keygen -t ed25519 -C "你的邮箱"      # 一路回车
cat ~/.ssh/id_ed25519.pub                 # 复制输出
# 粘到 https://github.com/settings/keys → New SSH key
ssh -T git@github.com                     # 看到 "Hi <用户名>!" 就成功
```

---

## 2. 关联远端并推送

```bash
cd /home/liqingchen/project/cbe-release

# 把 <你的用户名> 换成实际的
git remote add origin https://github.com/<你的用户名>/cbe.git   # 或 git@github.com:<用户名>/cbe.git

git push -u origin main
```

**验证**：打开 `https://github.com/<用户名>/cbe`，应该看到 README 渲染出来、
两个侧栏徽章（Cite this repository / MIT license）。

**如果 push 被拒**：
- `Support for password authentication was removed` → 你在用账号密码，改用 token（步骤 1A）
- `remote origin already exists` → `git remote set-url origin <新地址>`
- 网络超时 → 你机器上走代理，给 git 也配上：
  `git config --global http.proxy http://127.0.0.1:7890`（端口按你的代理改）

---

## 3. 把 17.5 GiB 索引放到 Hugging Face

你的机器上**已经装了 `hf` CLI**，不用额外安装。

```bash
hf auth login          # 粘贴 https://huggingface.co/settings/tokens 的 token（write 权限）

# 建数据集仓库（私有也可以，但建议 public 方便审稿人）
hf repo create cbe-indexes --repo-type dataset
```

然后把索引按 `<语料>/<文件名>` 的结构传上去。**建议一个语料一个命令、失败可重试**：

```bash
# MPtrj（3.0 GB，单文件）
hf upload <你的HF用户名>/cbe-indexes \
  /home/liqingchen/project/CBE/examples/mptrj_index_all/index.faiss \
  mptrj/index.faiss --repo-type dataset

# Alexandria（9.5 GB，4 个分片）
for i in 000 001 002 003; do
  hf upload <你的HF用户名>/cbe-indexes \
    /home/liqingchen/project/alexandria_index_full/shard_$i.faiss \
    alexandria/shard_$i.faiss --repo-type dataset
done

# OMat24（5.2 GB，5 个分片）
for i in 000 001 002 003 004; do
  hf upload <你的HF用户名>/cbe-indexes \
    /home/liqingchen/project/omat24_index/shard_$i.faiss \
    omat24/shard_$i.faiss --repo-type dataset
done
```

> 目录结构必须是 `<语料>/<文件名>`，因为下载脚本按 `f"{corpus}/{name}"` 取文件。
> `pca.npz` 和 `meta.json` **不用传**——它们在 git 仓库里。

**传完后改仓库里的两处占位符**（否则下载脚本不知道去哪拿）：

```bash
cd /home/liqingchen/project/cbe-release
# 1) 下载脚本的默认仓库名
sed -i 's|<account>/cbe-indexes|<你的HF用户名>/cbe-indexes|' examples/index/download_pretrained.py
# 2) README 与稿件里的仓库地址
sed -i 's|https://github.com/<account>/cbe|https://github.com/<你的用户名>/cbe|g' README.md pyproject.toml CITATION.cff
grep -rn "<account>" . --include="*.py" --include="*.md" --include="*.toml" --include="*.cff"
# 上面这条应该没有输出（说明占位符都填完了）
git commit -am "Point downloads at the published index repository"
git push
```

---

## 4. 拿 Zenodo DOI（论文里承诺的）

1. 用 GitHub 账号登录 https://zenodo.org → *Settings* → *GitHub* → 找到 `cbe` → 打开开关
2. 回 GitHub 建一个 **Release**（Zenodo 只认 release）：
   ```bash
   cd /home/liqingchen/project/cbe-release
   git tag -a v0.6.0 -m "CBE 0.6.0 - first release accompanying the paper"
   git push origin v0.6.0
   ```
   然后 `https://github.com/<用户名>/cbe/releases/new` → 选 `v0.6.0` → Publish
3. 几分钟后 Zenodo 的 *Upload* 页会出现这条记录，**给它一个 DOI**（记住 "Version DOI" 和
   "Concept DOI"：论文里引用具体版本用前者，长期指引用后者）
4. 在 Zenodo 记录里补上 Hugging Face 索引的链接（编辑 → Related identifiers → *is supplemented by*）

**数据本身的 DOI**：索引 17.5 GiB 超过 Zenodo 单文件推荐值，但可以给
Hugging Face 仓库单独出一份说明页（HF 仓库自带 DOI 功能：仓库 Settings → DOI），
或者把 `CHECKSUMS.json` + 下载说明作为一个小的 Zenodo 记录归档，DOI 指向校验清单。

---

## 5. 把 DOI 和作者信息填回论文

```bash
cd /home/liqingchen/project/DOC
# 逐处替换（5 个占位符）
grep -n "First Author\|\[account\]\|DOI to be assigned\|Zenodo DOI\|\[institution\]" jctc.tex
```

对照表：

| 稿件里的占位符 | 替换成 |
|---|---|
| `\author{First Author}` / `\author{Second Author}` | 真实姓名 |
| `Department of Physics, University, City, Country` | 真实单位 |
| `corresponding.author@university.edu`（2 处） | 真实通讯邮箱 |
| `\texttt{[institution]}`（致谢） | 提供算力的单位 |
| `\texttt{[Zenodo DOI]}`（Data / Code availability，2 处） | 上一步的 DOI |
| `\texttt{https://github.com/[account]/cbe}` | 你的仓库地址 |

改完重新编译投稿版：

```bash
cd /home/liqingchen/project/DOC
XDG_CACHE_HOME=/home/liqingchen/project/.cache ../.tools/tectonic --keep-logs jctc_submission.tex
# 检查 0 告警
grep -c 'Overfull\|undefined' jctc_submission.log
```

> **注意**：`Data availability` 里写的是"索引在 Hugging Face + Zenodo 归档"，
> 所以 DOI 填 Zenodo 的、HF 链接也要出现在 Data availability 正文里（确认一下这句话还在）。

---

## 投 JCTC 之前的最后检查清单

- [ ] `git push` 成功，仓库页面能打开，README 里的命令能照着跑
- [ ] `pretrained/indexes/*/pca.npz` 三个文件**在 GitHub 上确实存在**（这是索引能否使用的关键）
- [ ] HF 上的七个 `.faiss` 文件都能下载（拿其中最小的 `omat24/shard_004.faiss` 试一次）
- [ ] 在一台**别的机器**（或干净的 conda 环境）上验证一遍：
      `pip install -e ".[index]"` → `download_pretrained.py --corpus mptrj --skip-verify` →
      `verify_index.py`（这一步能发现 90% 的发布事故）
- [ ] 论文里所有占位符清零：`grep -rn "\[account\]\|DOI to be assigned\|First Author" DOC/*.tex`
- [ ] `jctc_submission.pdf` 34 页 0 告警，`si.pdf` 15 页 0 告警
- [ ] JCTC 投稿系统要求：正文 PDF + SI PDF + cover letter（可另写）+
      可能的 `Supporting Information` 单独文件

---

## 常见问题

**Q: 为什么不上传索引到 GitHub？**
A: 单个文件 2.94 GB 超过 GitHub 的 2 GB 单文件硬限；17.5 GB 也远超仓库 5 GB 建议上限。
用 git-lfs 的话，10 GB 文件被下载 20 次就爆免费额度，而且普通 `git clone` 拿不到数据。

**Q: 以后改代码怎么更新？**
```bash
cd /home/liqingchen/project/cbe-release
git add -A && git commit -m "说明这次改了什么" && git push
```
索引不用重传（HF 上的是独立版本）。

**Q: 想改索引怎么办？**
重传同名文件即可，但**必须同步更新 `pretrained/CHECKSUMS.json` 里对应的 sha256**，
否则用户下载后会校验失败。改完 `git commit` 一起推。

**Q: HF 仓库想设成私有？**
可以，但审稿人无法访问，Data availability 里就不能写"公开可用"。建议 public。
