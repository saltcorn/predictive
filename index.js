const Field = require("@saltcorn/data/models/field");
const Table = require("@saltcorn/data/models/table");
const Form = require("@saltcorn/data/models/form");
const Workflow = require("@saltcorn/data/models/workflow");
const FieldRepeat = require("@saltcorn/data/models/fieldrepeat");
const {
  eval_expression,
  jsexprToWhere,
} = require("@saltcorn/data/models/expression");
const fs = require("fs");
const fsp = fs.promises;
const path = require("path");
const {
  field_picker_fields,
  picked_fields_to_query,
  stateFieldsToWhere,
  stateFieldsToQuery,
  readState,
} = require("@saltcorn/data/plugin-helper");
const {
  get_predictor,
  write_csv,
  run_jupyter_model,
} = require("@saltcorn/data/model-helper");
const { mergeIntoWhere } = require("@saltcorn/data/utils");
const {
  text,
  div,
  pre,
  code,
  h5,
  table,
  tr,
  td,
  th,
  tbody,
} = require("@saltcorn/markup/tags");

const configuration_workflow = (req) =>
  new Workflow({
    steps: [
      {
        name: "Predictors",
        form: async (context) => {
          const table = await Table.findOne(
            context.table_id
              ? { id: context.table_id }
              : { name: context.exttable_name }
          );
          //console.log(context);
          const field_picker_repeat = await field_picker_fields({
            table,
            viewname: context.viewname,
            req,
            no_fieldviews: true,
          });

          const type_pick = field_picker_repeat.find((f) => f.name === "type");
          type_pick.attributes.options = type_pick.attributes.options.filter(
            ({ name }) =>
              ["Field", "JoinField", "Aggregation", "FormulaValue"].includes(
                name
              )
          );

          const use_field_picker_repeat = field_picker_repeat.filter(
            (f) =>
              !["state_field", "col_width", "col_width_units"].includes(f.name)
          );

          return new Form({
            fields: [
              new FieldRepeat({
                name: "columns",
                fancyMenuEditor: true,
                fields: use_field_picker_repeat,
              }),
            ],
          });
        },
      },
      {
        name: "Outcome variable",
        form: async (context) => {
          const table = await Table.findOne(
            context.table_id
              ? { id: context.table_id }
              : { name: context.exttable_name }
          );
          //console.log(context);
          const field_options = table.fields.filter((f) =>
            ["Float", "Int"].includes(f.type?.name)
          );
          return new Form({
            fields: [
              {
                name: "outcome_field",
                label: "Outcome field",
                subfield: "The variable that is to be predicted",
                type: "String",
                attributes: {
                  options: field_options,
                },
              },
              {
                name: "include_fml",
                label: req.__("Row inclusion formula"),
                class: "validate-expression",
                sublabel:
                  req.__("Only include rows where this formula is true. ") +
                  req.__("In scope:") +
                  " " +
                  [
                    ...table.fields.map((f) => f.name),
                    "user",
                    "year",
                    "month",
                    "day",
                    "today()",
                  ]
                    .map((s) => `<code>${s}</code>`)
                    .join(", "),
                type: "String",
                help: {
                  topic: "Inclusion Formula",
                  context: { table_name: table.name },
                },
              },
              {
                name: "split_test_train",
                label: "Split dataset",
                sublabel: "Split into training and testing datasets.",
                type: "Bool",
              },
              {
                name: "regression_model",
                label: "Regression model",
                type: "String",
                required: true,
                attributes: {
                  options: [
                    "Linear",
                    "Ridge",
                    "Lasso",
                    "Random Forest",
                    "Support Vector Machine",
                    "Partial Least Squares",
                    "Custom Python Code",
                  ],
                },
              },              
            ],
          });
        },
      },
    ],
  });

let regr_pred_code = get_predictor(path.join(__dirname, "Regression.ipynb"));

module.exports = {
  sc_plugin_api_version: 1,
  plugin_name: "sklearn-regression",
  modelpatterns: {
    RegressionModel: {
      configuration_workflow,
      hyperparameter_fields: ({ table, configuration }) => {
        switch (configuration?.regression_model) {
          case "Ridge":
            return [
              {
                name: "regularization",
                label: "L2 Regularization parameter",
                type: "Float",
                attributes: { min: 0 },
              },
            ];
          case "Lasso":
            return [
              {
                name: "regularization",
                label: "L1 Regularization parameter",
                type: "Float",
                attributes: { min: 0 },
              },
            ];
          case "Partial Least Squares":
            return [
              {
                name: "components",
                label: "Number of components to keep",
                type: "Integer",
                attributes: { min: 1 },
              },
            ];
          case "Random Forest":
            return [
              {
                name: "pca",
                label: "PCA preprocess",
                type: "Bool",
              },
              {
                name: "components",
                label: "Number of components to keep",
                type: "Integer",
                attributes: { min: 1 },
                showIf: { pca: true },
              },
            ];
          case "Support Vector Machine":
            return [
              {
                name: "kernel",
                label: "Kernel",
                type: "String",
                required: true,
                attributes: { options: ["linear", "poly", "rbf", "sigmoid"] },
              },
              {
                name: "C",
                label: "C",
                sublabel: "Regularization parameter",
                type: "Float",
                attributes: { min: 0 },
                default: 1.0,
              },
              {
                name: "degree",
                label: "Degree",
                type: "Integer",
                showIf: { kernel: "poly" },
              },
            ];
          default:
            return [];
        }
      },
      metrics: { R2: { lowerIsBetter: true } },
      prediction_outputs: ({ table, configuration }) => {
        return [
          { name: `${configuration.outcome_field}_prediction`, type: "Float" },
        ];
      },
      renderModel: ({ configuration }) =>
        table(
          tbody(
            tr(
              th({ class: "pe-2" }, "Model"),
              td(configuration.regression_model)
            ),
            tr(
              th({ class: "pe-2" }, "Predictors"),
              td(
                configuration.columns
                  .map(
                    (c) =>
                      c.field_name ||
                      c.join_field ||
                      c.agg_relation ||
                      c.formula
                  )
                  .join(", ")
              )
            ),
            tr(
              th({ class: "pe-2" }, "Outcome field"),
              td(configuration.outcome_field)
            ),
            configuration.include_fml &&
              tr(
                th({ class: "pe-2" }, "Row inclusion formula"),
                td(configuration.include_fml)
              ),
            tr(
              th({ class: "pe-2" }, "Split dataset"),
              td((!!configuration.split_test_train).toString())
            ),
            configuration.regression_model === "Custom Python Code" &&
              tr(
                th({ class: "align-top pe-2" }, "Model code"),
                td({ class: "align-top" }, pre(code(configuration.model_code)))
              )
          )
        ),

      train: async ({ table, configuration, hyperparameters, state }) => {
        const { columns, outcome_field } = configuration;
        //write data to CSV
        const fields = table.fields;

        readState(state, fields);
        const { joinFields, aggregations } = picked_fields_to_query(
          columns,
          fields
        );
        const where = await stateFieldsToWhere({ fields, state, table });
        if (configuration.include_fml) {
          let where1 = jsexprToWhere(configuration.include_fml, {}, fields);
          mergeIntoWhere(where, where1 || {});
        }
        //console.log("getting rows");
        //{ and: [{ not: { id: null } }, { not: { x: null } }] }
        //console.log(columns);

        where.and = columns
          .filter((c) => c.type === "Field")
          .map((c) => ({ not: { [c.field_name]: null } }));
        console.log("where", JSON.stringify(where, null, 2));

        //throw new Error("jkopi");
        let rows = await table.getJoinedRows({
          where,
          joinFields,
          aggregations,
        });
        console.log("writing csv");

        await write_csv(
          rows,
          [...columns, { type: "Field", field_name: outcome_field }],
          fields,
          "/tmp/scdata.csv"
        );
        console.log("running model");

        return await run_jupyter_model({
          configuration,
          hyperparameters,
          csvPath: "/tmp/scdata.csv",
          ipynbPath: path.join(__dirname, "Regression.ipynb"),
        });
      },
      predict: async ({
        id, //instance id
        model: {
          configuration: { columns, outcome_field },
          table_id,
        },
        hyperparameters,
        fit_object,
        rows,
      }) => {
        await fsp.writeFile("/tmp/scanomallymodel" + id, fit_object);
        const table = Table.findOne({ id: table_id });
        const rnd = Math.round(Math.random() * 10000);
        await write_csv(rows, columns, table.fields, `/tmp/scdata${rnd}.csv`);
        await eval("python.ex`" + regr_pred_code + "`");
        const predicts =
          await python`predict('/tmp/scanomallymodel'+str(${id}), ${`/tmp/scdata${rnd}.csv`})`;
        return rows.map((r, ix) => ({
          [`${outcome_field}_prediction`]: predicts.yhat[ix],
        }));
      },
    },
  },
};
