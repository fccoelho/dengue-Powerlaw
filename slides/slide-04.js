const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'content',
  index: 4,
  title: 'Data Sources'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.bg };

  slide.addText("Data Sources", {
    x: 0.5, y: 0.3, w: 9, h: 0.8,
    fontSize: 32, fontFace: "Arial",
    color: theme.primary, bold: true
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.0, w: 2, h: 0.05,
    fill: { color: theme.accent }
  });

  const dataSources = [
    {
      title: "Infodengue API",
      items: [
        "Weekly estimated case counts (casos_est)",
        "15+ years of data (2010-present)",
        "5,000+ Brazilian municipalities",
        "Climate variables (temperature, humidity)"
      ]
    },
    {
      title: "Episcanner Data",
      items: [
        "Annual epidemic characteristics",
        "Total cases per epidemic",
        "Epidemic duration (weeks)",
        "Peak week timing",
        "Basic reproduction number (R₀)"
      ]
    }
  ];

  dataSources.forEach((source, i) => {
    const x = 0.5 + i * 4.7;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x, y: 1.3, w: 4.4, h: 3.2,
      fill: { color: "FFFFFF" },
      line: { color: theme.accent, width: 2 },
      rectRadius: 0.1
    });
    slide.addText(source.title, {
      x: x + 0.2, y: 1.45, w: 4, h: 0.4,
      fontSize: 18, fontFace: "Arial",
      color: theme.primary, bold: true
    });
    slide.addText(
      source.items.map((item, idx) => ({
        text: item,
        options: { bullet: true, breakLine: idx < source.items.length - 1 }
      })),
      {
        x: x + 0.2, y: 1.9, w: 4, h: 2.4,
        fontSize: 13, fontFace: "Arial",
        color: theme.secondary
      }
    );
  });

  slide.addText("Data integrated via Mosqlimate Platform", {
    x: 0.5, y: 4.7, w: 9, h: 0.4,
    fontSize: 12, fontFace: "Arial",
    color: theme.accent, italic: true, align: "center"
  });

  slide.addShape(pres.shapes.OVAL, {
    x: 9.3, y: 5.1, w: 0.4, h: 0.4,
    fill: { color: theme.accent }
  });
  slide.addText("4", {
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
  pres.writeFile({ fileName: "slide-04-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
