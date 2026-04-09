# Learning Modules Overview

## Purpose
This folder contains **pure learning materials** - theory, math, and toy examples to build deep understanding of Information Retrieval concepts. **Completely separate from project implementation.**

---

## Structure

```text
learning/
├── README.md                                 # Main entry point
├── modules/
│   ├── README.md                             # This file - quick reference
│   ├── 01_text_processing_and_indexing/
│   ├── 02_ranked_retrieval/
│   └── 03_evaluation/
└── raw_materials/
    └── week_*/                               # Course slides, labs, and reference files
```

---

## Learning Philosophy

### **Theory First, Implementation Second**
1. **Read** `theory.md` - Understand concepts deeply
2. **Derive** formulas by hand on paper
3. **Run** the notebook examples - See it work on simple data
4. **Practice** `exercises.ipynb` - Test understanding
5. **Review** class materials (slides + labs)
6. **Then** apply to project (in `notebooks/` and `src/`)

### **Deep Understanding Over Quick Results**
- Don't memorize - understand WHY
- Derive all formulas from first principles
- Work examples by hand before coding
- Build intuition on toy data (5-10 docs)
- **Only then** scale to real data (1.3M entities)

---

## Module Progression

### **Module 1: Indexing & TF-IDF** (Foundation)
**Duration:** 4-5 days
**Manning Book:** Chapters 1, 2, 6
**Focus:**
- Inverted index structure
- Tokenization, stemming, lemmatization
- TF-IDF formula and intuition
- Vector Space Model
- Cosine similarity

**Success:** Can build inverted index and calculate TF-IDF by hand

---

### **Module 2: Classical IR** ⭐ **MOST IMPORTANT**
**Duration:** 7-8 days (spend 2 days just on BM25!)
**Manning Book:** Chapters 1, 6, 11
**Focus:**
- Boolean model
- Vector Space Model (deep)
- Binary Independence Model (probabilistic IR)
- **BM25 (the star)** - understand every component
- BM25F (multi-field)

**Success:** Can derive BM25 formula and explain every parameter

---

### **Module 3: Evaluation**
**Theory:** [03_evaluation/theory.md](03_evaluation/theory.md)

**Duration:** 4-5 days
**Manning Book:** Chapter 8
**Focus:**
- Precision & Recall
- Mean Average Precision (MAP)
- nDCG (graded relevance)
- MRR, Success@K

**Success:** Can calculate MAP and nDCG by hand

---

### **Module 4: Dense Retrieval** (Optional/Lightweight)
**Duration:** 2-3 days
**Manning Book:** N/A (too modern)
**Focus:**
- Sentence embeddings
- Semantic similarity
- Hybrid retrieval (sparse + dense)
- Reciprocal Rank Fusion (RRF)

**Success:** Understand concepts to help teammate (Marek)

---

## Relationship to Project

### **Learning Modules** (this folder)
- Pure theory and toy examples
- Small, simple datasets (5-10 documents)
- Focus on understanding concepts
- No time pressure
- Experiment freely

### **Project Implementation** (`notebooks/` and `src/`)
- Apply learned concepts
- Real OpenSanctions data (1.3M entities)
- Production-quality code
- Team collaboration
- Assignment deliverables

**Flow:**
```
Learn in learning/ → Apply in notebooks/ → Productionize in src/
```

---

## Time Investment

| Module | Days | What You Get |
|--------|------|--------------|
| Module 1 | 4-5 | Foundation: indexing, TF-IDF, VSM |
| Module 2 | 7-8 | Classical IR mastery (esp. BM25) |
| Module 3 | 4-5 | Evaluation expertise |
| Module 4 | 2-3 | Modern IR awareness (optional) |
| **Total** | **17-21 days** | **Deep IR understanding** |

**Worth it?** Absolutely! You'll:
- Understand concepts deeply (not superficially)
- Be able to debug issues
- Make informed design decisions
- Explain to others clearly
- Excel in exams and interviews

---

## How to Use This Folder

### **Starting a New Module:**
1. Start with the module's `theory.md`
2. Read the part-by-part breakdown
3. Run the accompanying notebook or toy example
4. Work through sequentially

### **While Learning:**
1. Keep a study journal in your own notes
2. Derive formulas on paper
3. Work examples by hand
4. Verify with toy code
5. Ask questions (lectures, office hours)

### **After Completing Module:**
1. Review the success criteria below
2. Can you check all boxes?
3. If not, review weak areas
4. **Then** move to project implementation

---

## Success Criteria Summary

### **After Module 1:**
- [ ] Can explain inverted index
- [ ] Can derive TF-IDF formula
- [ ] Can calculate cosine similarity by hand
- [ ] Built working search engine on toy data

### **After Module 2:**
- [ ] **Can derive BM25 formula from scratch** ⭐
- [ ] **Understand k₁ and b parameters deeply** ⭐
- [ ] Can implement all classical models
- [ ] Know when to use which model

### **After Module 3:**
- [ ] **Can calculate MAP by hand** ⭐
- [ ] **Can calculate nDCG by hand** ⭐
- [ ] Know which metric for which task
- [ ] Can interpret evaluation results

### **After Module 4:**
- [ ] Understand dense vs sparse retrieval
- [ ] Can explain when dense helps
- [ ] Can assist with hybrid systems

---

## Key Principles

### **1. Math Mastery**
- Don't skip derivations
- Work examples by hand
- Understand every symbol
- Ask "why?" for every formula

### **2. Toy Examples Are Essential**
- Build intuition on simple data first
- Can trace execution by hand
- Debug easily
- See exactly why it works

### **3. Manning Book Is Your Bible**
- Read assigned chapters fully
- Work through examples in book
- Do end-of-chapter exercises
- Keep it as reference

### **4. Class Materials Connect**
- Slides reinforce concepts
- Labs show practical implementation
- Compare with your understanding
- Identify gaps

### **5. Project Comes After**
- Don't rush to project
- Build solid foundation first
- Deep understanding enables fast implementation
- Better code quality with understanding

---

## Common Questions

**Q: Can I skip toy examples and go straight to project?**
A: You can, but you'll struggle. Toy examples build intuition that makes project much easier.

**Q: Do I really need to derive formulas by hand?**
A: Yes! Derivations reveal the "why" behind formulas. You'll remember and understand deeply.

**Q: How much time should I spend per module?**
A: Follow the day estimates in this overview. Module 2 (BM25) needs 7-8 days - don't rush it!

**Q: What if I don't understand something?**
A:
1. Re-read Manning chapter
2. Work more examples by hand
3. Check online resources
4. Ask in lectures/office hours
5. Discuss with team

**Q: Can I work on project while learning?**
A: Better to finish each module first, then apply. But you can alternate if you prefer.

---

## Resources

### **Primary:**
- Manning's "Introduction to Information Retrieval" (the book in class_materials/)
- This overview and the module `theory.md` files
- Module theory.md files

### **Supplementary:**
- Course slides (class_materials/week_X/)
- Labs (class_materials/week_X/)
- Online: Stanford CS276 materials
- Papers (referenced in theory files)

### **Tools:**
- Python 3.13+
- Jupyter notebooks
- Libraries: nltk, spaCy, scikit-learn, rank-bm25, sentence-transformers

---

## Next Steps

1. ✅ Read this overview
2. 📚 Start Module 1, Part 1 (Boolean Retrieval)
3. 📝 Keep study journal
4. 🚀 Build deep understanding!

---

## Remember

> "Give me six hours to chop down a tree and I will spend the first four sharpening the axe."
> - Abraham Lincoln

**Invest time in learning modules → Project becomes much easier** 🎯

---

**Ready to become an IR expert? Start with Module 1!** 🚀
