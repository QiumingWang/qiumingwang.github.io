---
title: test_collection
time: 2024-01-06 23:59:00 +8000
categories:
  - test
tags: null
---

# test

test somethings



```yaml
# include:
# #   - ".htaccess"
#   - "_pages"
# #   - "_post"

jekyll-archives:
  enabled: [categories, tags]
  layouts:
    category: category
    tag: tag
  permalinks:
    tag: /tags/:name/
    category: /categories/:name/

# collections:
#   pages:
#     output: true
#     permalink: /_pages/:path
  # ethos:
  #   output: true
  #   permalink: /:collections/:path/
```