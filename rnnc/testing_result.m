
data = importdata('sample.txt');
txt = fopen('finalResult.txt', 'w');


idx = data(:,1);
pro = data(:,2:end);

for i = 1:length(idx)
	[prbMax, id] = min(pro(i,:));
	answer = 'a';
	switch id
		case 1
			answer = 'a';
		case 2
			answer = 'b';
		case 3
			answer = 'c';
		case 4
			answer = 'd';
		case 5
			answer = 'e';
	end
	
	fprintf(txt, [int2str(i) ',' answer '\n']);
	
end