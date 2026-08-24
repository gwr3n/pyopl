# Save Exemplar Roadmap

## Goal

Add a `Save Exemplar...` command to the File menu immediately after
`Export Model...`. The command will save the current model, current data, and
complete working-directory `.pyopl_session` content as a RAG exemplar triplet
under `./opl_models`.

## Confirmed Behavior

- Prompt for a relative exemplar path beneath the working-directory
  `opl_models` folder.
- Allow nested paths. For an entry such as `routing/fleet`, create
  `./opl_models/routing/fleet/` and use the leaf name for the triplet:
  `fleet.mod`, `fleet.dat`, and `fleet.txt`.
- Reject absolute paths, empty path components, `.` or `..` components, and
  paths that resolve outside `./opl_models`.
- Refuse an exemplar path whose destination folder already exists and ask the
  user to choose another name; never overwrite an existing exemplar.
- Create the working-directory `opl_models` root and any requested parent
  folders when they do not exist.
- Save the model and data directly from the current editor buffers without
  changing their active file paths or saved/dirty state.
- Invoke session persistence first, then copy the entire UTF-8
  `.pyopl_session` file content into the `.txt` description file unchanged.
- Warn and leave no partial exemplar folder if the model, data, or session
  content cannot be collected or written.

## Implementation Steps

1. Insert the File menu command immediately after `Export Model...`.
2. Add a small relative-path validator that supports nested paths while
   containing all output beneath `Path.cwd() / "opl_models"`.
3. Add the command handler to prompt for a name, validate it, reject existing
   destinations, persist/read `.pyopl_session`, and collect editor contents.
4. Create the exemplar in a temporary sibling folder and rename it into place
   only after all three files are written successfully.
5. Report success in the IDE status bar and report validation or I/O failures
   through focused dialogs.
6. Add `unittest` coverage for menu placement, simple and nested paths,
   triplet contents, root creation, cancellation, traversal rejection,
   existing-folder refusal, and cleanup after write failure.
7. Update `CHANGELOG.md`, run the focused IDE tests, and check diagnostics and
   formatting for all touched files.

## Acceptance Criteria

- The File menu shows `Save Exemplar...` directly after `Export Model...`.
- Saving `routing/fleet` produces
  `opl_models/routing/fleet/fleet.{mod,dat,txt}`.
- The `.mod` and `.dat` files exactly reflect the current editor buffers.
- The `.txt` file exactly matches the complete persisted `.pyopl_session`
  content at save time.
- Existing exemplar folders and paths outside `./opl_models` are never
  overwritten.
- A failed save leaves no incomplete destination folder.