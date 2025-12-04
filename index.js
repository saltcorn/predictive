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

const {
  linearModel,
  preprocessing,
  dataset,
  ensemble,
  modelSelection,
  metrics,
  pipeline,
  cluster,
  decomposition,
  naiveBayes,
  neighbors,
  dataFrame,
  coreBindings,
} = require("@saltcorn/smartcore-js");
let { RidgeRegressionF64F64 } = coreBindings;

let {
  LogisticRegression,
  LinearRegression,
  RidgeRegression,
  Lasso,
  ElasticNet,
} = linearModel;
let { RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor } =
  ensemble;
let { StandardScaler, OneHotEncoder } = preprocessing;
let { KMeans, DBSCAN } = cluster;
let { PCA, SVD } = decomposition;
let { BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB } = naiveBayes;
let { KNNClassifier, KNNRegressor } = neighbors;
let { loadIris, loadBoston, loadBreastCancer, loadDiabetes, loadDigits } =
  dataset;
let { trainTestSplit } = modelSelection;
let { accuracyScore, DistanceType } = metrics;
let { makePipeline } = pipeline;
const { DataFrame } = dataFrame;

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
                  ],
                },
              },
            ],
          });
        },
      },
      {
        name: "Preprocessing steps",
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
              new FieldRepeat({
                name: "preprocessors",
                fields: [
                  {
                    name: "preproctype",
                    label: "Type",
                    type: "String",
                    required: true,
                    attributes: { options: ["Standard scaler", "PCA"] },
                  },
                  {
                    name: "pca_ncomponents",
                    label: "Number of components",
                    type: "Integer",
                    required: true,
                    showIf: { preproctype: "PCA" },
                  },
                  {
                    name: "pca_columns",
                    label: "Predictors to include in PCA",
                    type: "String",
                    sublabel:
                      "Comma separated list of variable names. Leave blank to transform all predictors",
                    showIf: { preproctype: "PCA" },
                  },
                ],
              }),
            ],
          });
        },
      },
    ],
  });

module.exports = {
  sc_plugin_api_version: 1,
  plugin_name: "predictive",
  modelpatterns: {
    SupervisedPredictionModel: {
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

        const { df, float_columns } = rows_to_df({
          rows,
          configuration,
          table,
        });
        const y_key = configuration.outcome_field;
        // console.log('Selected: ', columns)

        let pipe = makePipeline(
          [
            //new StandardScaler(),
            new PCA({ nComponents: 3, columns: float_columns }),
            new RidgeRegression({ alpha: hyperparameters.regularization }),
          ],
          {
            verbose: true,
          }
        );
        const y = new Float64Array(rows.map((r) => r[y_key]));

        pipe.fit(df, y);
        let score = calcr2(pipe.predict(df), y);

        const fit_object = serialisePipe(pipe);
        return { fit_object, metric_values: { R2: score } };
      },
      predict: async ({
        id, //instance id
        model: { configuration, table_id },
        hyperparameters,
        fit_object,
        rows,
      }) => {
        const pipe = deserialisePipe(fit_object);
        const { df } = rows_to_df({ rows, configuration, table });
        const yhats = Array.from(pipe.predict(df));
        return yhats.map((yhat) => ({
          [`${configuration.outcome_field}_prediction`]: yhat,
        }));
      },
    },
  },
};
function calcr2(y, f) {
  let sum = 0;
  for (let i = 0; i < y.length; i++) {
    sum += y[i];
  }
  let ymean = sum / y.length;
  let ssres = 0,
    sstot = 0;
  for (let i = 0; i < y.length; i++) {
    sstot += Math.pow(y[i] - ymean, 2);
    ssres += Math.pow(y[i] - f[i], 2);
  }

  return 1 - ssres / sstot;
}
const rows_to_df = ({ rows, configuration, table }) => {
  const float_columns = [];
  const cols = configuration.columns.map((c) => {
    if (c.type !== "Field") return c;
    const field = table.getField(c.field_name);
    const isBool = field?.type?.name === "Bool";
    if (!isBool) float_columns.push(c.field_name);
    let maxDims;
    if (field.type.name === "FloatArray") {
      const dims = rows.map((r) => r[c.field_name].length);
      maxDims = Math.max(...dims);
    }

    return { field, maxDims, isBool, ...c };
  });
  const df = new DataFrame(
    rows.map((r) => {
      const o = {};
      cols.forEach((c) => {
        switch (c.type) {
          case "FormulaValue":
            o[c.field_name] = eval_expression(c.formula, r);
            break;
          case "Field":
            if (c.field.type.name === "FloatArray") {
              for (let i = 0; i < c.maxDims; i++)
                o[c.field_name + i] = r[c.field_name][i];
            } else if (c.field.type.name === "PGVector") {
              const v = JSON.parse(r[c.field_name]);
              const dims = v.length;
              for (let i = 0; i < dims; i++) {
                o[c.field_name + i] = v[i];
              }
            } else {
              o[c.field_name] = c.isBool
                ? r[c.field_name]
                  ? 1.0
                  : 0.0
                : r[c.field_name];
            }
            break;
          default:
            break;
        }
      });
      return o;
    })
  );
  return { df, float_columns };
};
