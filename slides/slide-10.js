const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 10,
  title: 'Machine Learning Predictions'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  slide.addText("Machine Learning Predictions", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 32, fontFace: "Arial",
    color: theme.primary, bold: true
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.95, w: 2, h: 0.05,
    fill: { color: theme.accent }
  });

  slide.addText("Random Forest models leveraging power law features for epidemic forecasting", {
    x: 0.5, y: 1.15, w: 9, h: 0.4,
    fontSize: 14, fontFace: "Arial",
    color: theme.secondary, italic: true
  });

  const features = () => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.5, y: 1.6, w: 4.3, h: 2.5,
      fill: { color: "FFFFFF" },
      line: { color: theme.accent, width: 2 },
      rectRadius: 0.1
    });
    slide.addText("Feature Engineering", {
      x: 0.7, y: 1.75, w: 4, h: 0.4,
      fontSize: 14, fontFace: "Arial",
      color: theme.primary, bold: true
    });
    slide.addText([
      { text: "Lagged epidemiological metrics", options: { bullet: true, breakLine: true } },
      { text: "Historical α trends (slope)", options: { bullet: true, breakLine: true } },
      { text: "Quarterly climate data", options: { bullet: true, breakLine: true } },
      { text: "Population (log-transformed)", options: { bullet: true, breakLine: true } },
      { text: "Geographic encoding", options: { bullet: true } }
    ], {
      x: 0.7, y: 2.2, w: 4, h: 1.8,
      fontSize: 12, fontFace: "Arial",
      color: theme.secondary
    });
  };
  features();

  const targets = () => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 5.2, y: 1.6, w: 4.3, h: 2.5,
      fill: { color: theme.primary },
      rectRadius: 0.1
    });
    slide.addText("Prediction Targets", {
      x: 5.4, y: 1.75, w: 4, h: 0.4,
      fontSize: 14, fontFace: "Arial",
      color: "FFFFFF", bold: true
    });
    slide.addText([
      { text: "Total epidemic cases", options: { bullet: true, breakLine: true } },
      { text: "Epidemic duration", options: { bullet: true, breakLine: true } },
      { text: "Peak week timing", options: { bullet: true } }
    ], {
      x: 5.4, y: 2.2, w: 4, h: 1.2,
      fontSize: 12, fontFace: "Arial",
      color: theme.light
    });
    slide.addText("Log-transformed targets for total_cases handle order-of-magnitude variation", {
      x: 5.4, y: 3.4, w: 4, h: 0.5,
      fontSize: 10, fontFace: "Arial",
      color: theme.light, italic: true
    });
  };
  targets();

  const results = () => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.5, y: 4.25, w: 9, h: 1.1,
      fill: { color: theme.accent },
      rectRadius: 0.1
    });
    slide.addText("Key Result: ", {
      x: 0.7, y: 4.45, w: 1.3, h: 0.7,
      fontSize: 14, fontFace: "Arial",
      color: theme.primary, bold: true, valign: "middle"
    });
    slide.addText("Cities with lower α (heavier tails) show reduced predictability in standard models, creating \"prediction gaps\" where traditional forecasting fails — precisely when extreme outbreak potential is highest.", {
      x: 2.0, y: 4.35, w: 7.3, h: 0.9,
      fontSize: 12, fontFace: "Arial",
      color: theme.primary, valign: "middle"
    });
  };
  results();

  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent }
  });
  slide.addText("10", {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: "FFFFFF", bold: true,
    align: "center", valign: "middle"
  });

  return slide;
}

if (require.main === module) {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_16x9';
  const theme = {
    primary: "1A365D",
    secondary: "2C5282",
    accent: "4ECDC4",
    light: "E2E8F0",
    bg: "F7FAFC"
  };
  createSlide(pres, theme);
  pres.writeFile({ fileName: "slide-10-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
