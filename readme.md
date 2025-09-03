# About

# General notes on code architecture

* Most prompts are designed to return a valid JSON object, usually with schema `List[object]`

# The file structure of DocuCache

```shell
└── docucache/
    ├── metadata.db            <-- The SQLite database file
    ├── 1/                     <-- First paper's folder (ID from DB)
    │   ├── assets/
    │   └── tmp/
    │       └── 1706.03762.pdf
    ├── 2/
    │   ├── assets/
    │   └── tmp/
    │       └── 2203.02155.pdf
    └── 3/
        ├── assets/
        └── tmp/
            └── 2307.09288.pdf
```

