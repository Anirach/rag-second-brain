import type { HookHandler } from "../../src/hooks/hooks.js";
import * as fs from "fs";
import * as path from "path";

// In-memory message counter per session
const msgCount: Record<string, number> = {};
const SNAPSHOT_EVERY = 20;

const handler: HookHandler = async (event) => {
  if (event.type !== "message" || event.action !== "received") return;

  const sessionKey = event.sessionKey || "unknown";
  msgCount[sessionKey] = (msgCount[sessionKey] || 0) + 1;

  // Only snapshot every N messages
  if (msgCount[sessionKey] % SNAPSHOT_EVERY !== 0) return;

  try {
    const now = new Date();
    const bangkokOffset = 7 * 60;
    const bkk = new Date(now.getTime() + bangkokOffset * 60 * 1000);
    const dateStr = bkk.toISOString().slice(0, 10);
    const timeStr = bkk.toISOString().slice(11, 16);

    const workspaceDir = process.env.WORKSPACE_DIR || "/home/clawdbot/clawd";
    const memoryDir = path.join(workspaceDir, "memory");
    const memoryFile = path.join(memoryDir, `${dateStr}.md`);

    const content = event.context?.content || "";
    const from = event.context?.from || "unknown";
    const snippet = content.slice(0, 200).replace(/\n/g, " ");

    const entry = `\n### ${timeStr} BKK — Session Snapshot (msg #${msgCount[sessionKey]})\n` +
      `- Session: ${sessionKey}\n` +
      `- Last message from: ${from}\n` +
      `- Snippet: "${snippet}"\n` +
      `- Auto-snapshot: session active, ${msgCount[sessionKey]} messages exchanged\n`;

    fs.mkdirSync(memoryDir, { recursive: true });
    fs.appendFileSync(memoryFile, entry, "utf-8");

    console.log(`[session-snapshot] Snapshot saved to ${dateStr}.md (msg #${msgCount[sessionKey]})`);
  } catch (err) {
    console.error("[session-snapshot] Error:", err instanceof Error ? err.message : String(err));
  }
};

export default handler;
