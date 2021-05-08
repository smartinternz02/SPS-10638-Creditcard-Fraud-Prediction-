package project.ml;
import java.io.IOException;
import tech.tablesaw.api.Table;
public class card {
		public static void main(String args[])
		{
			System.out.println("credit card fraud prediction");
			try {
			Table creditcard = Table.read().csv("C:\\Users\\Sasi\\eclipse-workspace\\project.ml\\src\\main\\creditcard.csv");
			System.out.println(creditcard.shape());
			System.out.println(creditcard.first(5));
			System.out.println(creditcard.last(7));
			System.out.println(creditcard.structure());
			System.out.println(creditcard.summary());
			} catch (IOException a) 
			{
				a.printStackTrace();
			}
		}
}
	