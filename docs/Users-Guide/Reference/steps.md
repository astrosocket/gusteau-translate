---
title: Pipeline Steps
icon: lucide/waypoints
---


# Available Translation Pipeline Steps

::: gusteau_translate.translation_base.TranslationStep
    options:
        show_root_heading: True



## Translation-specific

### Attributes

::: gusteau_translate.translations.attribute_translation
    options:
        show_root_toc_entry: false
        heading_level: 4
        members:
            - AttributeLoadStep
            - AttributeSaveStep
            - SaveQuantityStep
            - SnapshotAttributeLoadStep

### Datasets

::: gusteau_translate.translations.dataset_translation
    options:
        show_root_toc_entry: false
        heading_level: 4
        members:
            - DatasetLoadStep
            - DatasetSaveStep
            - MakeParticleAttributesStep
            - MoveDatasetAttributesStep


### Groups

::: gusteau_translate.translations.group_translation
    options:
        show_root_toc_entry: false
        heading_level: 4
        members:
            - GroupLoadStep
            - GroupSaveStep
            - ProcessGroupChildrenStep


## Base Steps

::: gusteau_translate.base_steps
    options:
        show_root_toc_entry: false
        heading_level: 3

## Optional Steps

::: gusteau_translate.optional_steps
    options:
        show_root_toc_entry: false
        heading_level: 3


<script id="MathJax-script" src="https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
  window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };

  document$.subscribe(() => {
    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
  })
</script>