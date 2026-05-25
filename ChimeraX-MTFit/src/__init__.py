from chimerax.core.toolshed import BundleAPI


class _MTFitBundle(BundleAPI):
    """
    Entry point ChimeraX reads at startup.
    Registers the 'mtfit' command and (later) the GUI tool panel.
    """
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        if ci.name in ("mtfit", "mtfit setpath"):
            cmd.register_commands(logger)

    @staticmethod
    def start_tool(session, bi, ti):
        from . import tool
        return tool.MTFitTool(session, ti.name)

    @staticmethod
    def get_class(class_name):
        if class_name == "MTFitTool":
            from . import tool
            return tool.MTFitTool
        return None


bundle_api = _MTFitBundle()
