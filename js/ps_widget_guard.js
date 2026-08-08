/**
 * The prompt-widget write trap, isolated so it can be tested outside a browser.
 *
 * This is defensive only. The structural fix for the erasure bug is that prompts
 * live in their own widgets and every Asset Bin edit is a server-side merge; this
 * just turns a regression into a loud console error instead of silent data loss.
 *
 * It lives in its own module because the first version of it crashed workflow
 * loading, and a thing that can abort "Loading workflow data" needs a test. It
 * imports nothing, so Node can run it directly.
 *
 * Three failures it is written against, all seen or latent in that first version:
 *
 *   1. NOT IDEMPOTENT -- reloading a workflow re-ran the setup and called
 *      defineProperty on an already-trapped widget. Fixed by the __odProtected
 *      marker.
 *   2. NOT CAPABILITY-CHECKED -- if ComfyUI's own `value` descriptor is
 *      non-configurable, defineProperty throws "Cannot redefine property: value"
 *      on the very first call, aborting the load. Fixed by reading the descriptor
 *      and declining.
 *   3. IT REPLACED RATHER THAN WRAPPED -- swapping `value` for a plain closure
 *      variable severs the widget from any accessor ComfyUI had installed, which
 *      on those frontends means the typed prompt never reaches serialization.
 *      The user's text would be silently lost on save: the very bug the trap
 *      exists to prevent. Fixed by delegating to the previous descriptor.
 */

/** Is the Asset Bin currently performing a document write? */
export function isBinWriting(scope) {
  return Boolean((scope ?? globalThis).__odBinWriting);
}

/**
 * Install the trap on one widget. Safe to call repeatedly.
 * Returns "installed" | "already" | "skipped" | "failed" -- a status rather than
 * a throw, because no outcome here justifies stopping a workflow load.
 */
export function protectWidget(widget, name, options = {}) {
  const scope = options.scope ?? globalThis;
  const log = options.console ?? console;
  if (!widget) return "skipped";

  try {
    if (widget.__odProtected) return "already";

    const desc = Object.getOwnPropertyDescriptor(widget, "value");
    if (desc && desc.configurable === false) {
      log.debug?.(
        `[PulseStudio] ${name}.value is non-configurable; skipping the write trap. ` +
        `The structural protections are unaffected.`);
      return "skipped";
    }

    // Delegate to whatever was there before, so ComfyUI keeps owning the value.
    let get, set;
    if (desc && (desc.get || desc.set)) {
      get = desc.get ? desc.get.bind(widget) : () => undefined;
      const inner = desc.set ? desc.set.bind(widget) : () => {};
      set = inner;
    } else {
      let stored = desc ? desc.value : widget.value;
      get = () => stored;
      set = (v) => { stored = v; };
    }

    Object.defineProperty(widget, "value", {
      configurable: true,
      enumerable: desc ? desc.enumerable !== false : true,
      get,
      set(v) {
        if (isBinWriting(scope)) {
          log.error(
            `[PulseStudio] blocked a write to ${name} from the Asset Bin panel. ` +
            `Prompt widgets are user-owned; this is the erasure bug and it must not ` +
            `come back.`);
          return;
        }
        set(v);
      },
    });
    Object.defineProperty(widget, "__odProtected", {
      value: true, enumerable: false, configurable: true, writable: true,
    });
    return "installed";
  } catch (err) {
    // A trap is a nicety. Losing it costs a console warning; throwing here would
    // abort the user's workflow load.
    log.warn(`[PulseStudio] could not install the ${name} write trap:`, err);
    return "failed";
  }
}
