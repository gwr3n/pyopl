# Rhetor

**GenAI-assisted Integrated Modelling Environment for Mixed-Integer Linear Programming**

Rhetor is an Integrated Modelling Environment for modelling and solving mixed-integer linear programming problems. Rhetor supports a subset of the Optimisation Programming Language (OPL) syntax and it is built on [``pyopl``](https://github.com/gwr3n/pyopl), a Python library for parsing and solving OPL-like mathematical programming models using Gurobi or Scipy (Highs). Rhetor integrates GenAI features to support and automate the modelling process.

![Demo](https://github.com/gwr3n/rhetor/raw/main/sudoku.gif)

## Learn More

- [Project Website](https://gwr3n.github.io/rhetor)
- [Project Repository](https://github.com/gwr3n/rhetor)
- [PyOPL Repository](https://github.com/gwr3n/pyopl)
- [PyOPL User Guide](https://github.com/gwr3n/rhetor/blob/main/docs/PyOPL%20user%20guide.md)
- [Examples Overview](https://github.com/gwr3n/rhetor/blob/main/docs/PyOPL%20examples%20overview.md)
- [Sample PyOPL Models](https://github.com/gwr3n/rhetor/tree/main/opl_models)

## MCP access to the running IDE

When Rhetor is open, the Rhetor MCP server can read and replace the live model
and data editor contents. Add the server to an MCP client such as VS Code or
Claude Desktop using the Python executable from this environment:

```json
{
	"servers": {
		"Rhetor": {
			"type": "stdio",
			"command": "/path/to/venv/bin/python",
			"args": ["-m", "pyopl.rhetor_mcp"]
		}
	}
}
```

Start Rhetor before calling `read_ide_editors_tool` or
`write_ide_editors_tool`. The IDE bridge listens only on the loopback interface
and publishes per-launch connection credentials in the user's Rhetor config
directory. Writing through MCP updates the visible editors and marks the
contents as unsaved.