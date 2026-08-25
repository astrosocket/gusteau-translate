---
title: Translations
icon: lucide/languages
---

# Available Translations

## Base

::: gusteau_translate.translation_base.FieldTranslation
    options:
        show_root_heading: true
        heading_level: 3

## Attributes

::: gusteau_translate.translations.attribute_translation
    options:
        show_root_toc_entry: false
        heading_level: 3
        members:
            - AttributeTranslation
            - ConstantAttributeTranslation
            - SnapshotAttributeTranslation
            - FileContentsAttributeTranslation

## Datasets

::: gusteau_translate.translations.dataset_translation
    options:
        show_root_toc_entry: false
        heading_level: 3
        members: 
        - DatasetTranslation
        - ParticleDatasetTranslation
        - ConstantParticleDatasetTranslation

## Groups

::: gusteau_translate.translations.group_translation
    options:
        show_root_toc_entry: false
        heading_level: 3
        members:
            - GroupTranslation