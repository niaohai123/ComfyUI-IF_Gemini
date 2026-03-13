// gemini_node.js - Extension for IFGeminiPromptNode, IFGeminiImageGenNode
import { app } from "/scripts/app.js";

// ══════════════════════════════════════════════════════════════════════════
app.registerExtension({
    name: "Comfy.IFGeminiNode",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // ── IFGeminiPromptNode ───────────────────────────────────────────
        if (nodeData.name === "IFGeminiPromptNode") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }

                // Make prompt textarea larger
                const promptWidget = this.widgets.find(w => w.name === "prompt");
                if (promptWidget) {
                    promptWidget.computeSize = function(width) {
                        return [width, 120];
                    };
                }
            };
        }

        // ── IFGeminiImageGenNode ─────────────────────────────────────────
        if (nodeData.name === "IFGeminiImageGenNode") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }
            };
        }
    }
});