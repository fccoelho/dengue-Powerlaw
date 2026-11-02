const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 11,
  title: 'Implications'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  slide.addText("Implications & Applications", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 32, fontFace: "Arial",
    color: theme.primary, bold: true
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.95, w: 2, h: 0.05,
    fill: { color: theme.accent }
  });

  const implications = [
    {
      title: "Early Warning Systems",
      desc: "α parameters can serve as indicators of upcoming extreme event probability"
    },
    {
      title: "Resource Allocation",
      desc: "Regions with lower α require more surge capacity due to unpredictable extremes"
    },
    {
      title: "Climate Change Assessment",
      desc: "Tracking α trends reveals shifting epidemic regimes under climate change"
    },
    {
      title: "Model Improvement",
      desc: "Power law features improve prediction accuracy for extreme events"
    }
  ];

  implications.forEach((item, i) => {
    const y = 1.2 + i * 1.0;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.5, y: y, w: 9, h: 0.85,
      fill: { color: i % 2 === 0 ? "FFFFFF" : theme.light },
      line: { color: theme.accent, width: 1 },
      rectRadius: 0.08
    });
    slide.addText(item.title, {
      x: 0.7, y: y + 0.1, w: 8.6, h: 0.35,
      fontSize: 14, fontFace: "Arial",
      color: theme.primary, bold: true
    });
    slide.addText(item.desc, {
      x: 0.7, y: y + 0.45, w: 8.6, h: 0.35,
      fontSize: 12, fontFace: "Arial",
      color: theme.secondary
    });
  });

  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent }
  });
  slide.addText("11", {
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
  pres.writeFile({ fileName: "slide-11-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
