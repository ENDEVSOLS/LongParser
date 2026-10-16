"""Import-only external Docling SDK doubles; no conversion/model execution."""

from types import ModuleType


def install_docling_sdk_stubs(patch):
    import sys

    symbols = {
        "docling.datamodel.base_models": ["InputFormat"],
        "docling.datamodel.pipeline_options": ["PdfPipelineOptions", "TesseractCliOcrOptions"],
        "docling.document_converter": ["CsvFormatOption", "DocumentConverter", "ExcelFormatOption",
                                       "PdfFormatOption", "PowerpointFormatOption", "WordFormatOption"],
        "docling_core.transforms.chunker": ["HierarchicalChunker"],
        "docling_core.types.doc": ["ListItem", "PictureItem", "SectionHeaderItem", "TableItem", "TitleItem"],
    }
    modules = {}
    for name in symbols:
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            prefix = ".".join(parts[:end])
            if prefix not in modules:
                module = ModuleType(prefix)
                module.__path__ = []
                modules[prefix] = module
                patch.setitem(sys.modules, prefix, module)
    for name, names in symbols.items():
        for symbol in names:
            setattr(modules[name], symbol, type(symbol, (), {}))
